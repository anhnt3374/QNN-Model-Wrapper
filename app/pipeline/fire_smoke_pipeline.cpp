#include "pipeline/fire_smoke_pipeline.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <utility>

namespace pipeline {

FireSmokePipeline::FireSmokePipeline(
    models::FireSmokeDetectionModel& model,
    double targetInferenceFps,
    std::size_t temporalWindow,
    std::size_t confirmationCount
)
    : model_(
          model
      ),
      targetInferenceFps_(
          targetInferenceFps
      ),
      temporalWindow_(
          temporalWindow
      ),
      confirmationCount_(
          confirmationCount
      )
{
}

void FireSmokePipeline::setHtpExecutionGate(
    async_runtime::HtpExecutionGate* gate
) noexcept
{
    htpGate_ =
        gate;
}

bool FireSmokePipeline::configure(
    double sourceFps,
    int width,
    int height
)
{
    lastError_.clear();

    if (!model_.ready()) {

        lastError_ =
            "FireSmokeDetectionModel is not ready";

        return false;
    }

    if (!std::isfinite(
            sourceFps
        ) ||
        sourceFps <= 0.0) {

        lastError_ =
            "Invalid source FPS";

        return false;
    }

    if (!std::isfinite(
            targetInferenceFps_
        ) ||
        targetInferenceFps_ <= 0.0) {

        lastError_ =
            "Invalid FireSmoke inference FPS";

        return false;
    }

    if (width <= 0 ||
        height <= 0) {

        lastError_ =
            "Invalid video dimensions";

        return false;
    }

    if (temporalWindow_ == 0) {

        lastError_ =
            "Temporal window must be > 0";

        return false;
    }

    if (confirmationCount_ == 0 ||
        confirmationCount_ >
            temporalWindow_) {

        lastError_ =
            "Invalid FireSmoke confirmation count";

        return false;
    }

    if (targetInferenceFps_ >
        sourceFps) {

        targetInferenceFps_ =
            sourceFps;
    }

    inferencePeriodUs_ =
        static_cast<int64_t>(
            std::llround(
                1'000'000.0
                /
                targetInferenceFps_
            )
        );

    if (inferencePeriodUs_ <= 0) {

        lastError_ =
            "Invalid FireSmoke inference period";

        return false;
    }

    scheduleInitialized_ =
        false;

    nextInferenceTimestampUs_ =
        0;

    fireHistory_.clear();

    smokeHistory_.clear();

    fireConfirmed_ =
        false;

    smokeConfirmed_ =
        false;

    lastDetections_.clear();

    inferenceCount_ =
        0;

    return true;
}

bool FireSmokePipeline::shouldRunInference(
    const video::Frame& frame
)
{
    if (!scheduleInitialized_) {

        scheduleInitialized_ =
            true;

        nextInferenceTimestampUs_ =
            frame.captureTimestampUs
            +
            inferencePeriodUs_;

        return true;
    }

    if (frame.captureTimestampUs <
        nextInferenceTimestampUs_) {

        return false;
    }

    do {

        nextInferenceTimestampUs_ +=
            inferencePeriodUs_;

    } while (
        nextInferenceTimestampUs_
        <=
        frame.captureTimestampUs
    );

    return true;
}

bool FireSmokePipeline::process(
    const video::Frame& sourceFrame,
    video::Frame& renderFrame,
    std::vector<metrics::ModelMetrics>& modelMetrics
)
{
    lastError_.clear();

    if (sourceFrame.image.empty()) {

        lastError_ =
            "FireSmokePipeline received empty source frame";

        return false;
    }

    if (renderFrame.image.empty()) {

        lastError_ =
            "FireSmokePipeline received empty render frame";

        return false;
    }

    const bool runInference =
        shouldRunInference(
            sourceFrame
        );

    if (runInference) {

        metrics::ModelMetrics metric;

        metric.frameId =
            sourceFrame.frameId;

        metric.model =
            "firesmoke";

        const Clock::time_point modelStart =
            Clock::now();

        // =================================================
        // PREPROCESS
        // =================================================

        const Clock::time_point preprocessStart =
            Clock::now();

        if (!model_.preprocess(
                sourceFrame.image
            )) {

            lastError_ =
                "FireSmoke preprocess failed: "
                +
                model_.lastError();

            return false;
        }

        const Clock::time_point preprocessEnd =
            Clock::now();

        metric.preprocessMs =
            durationMs(
                preprocessStart,
                preprocessEnd
            );

        // =================================================
        // HTP
        // =================================================

        double htpWaitMs =
            0.0;

        double inferenceMs =
            0.0;

        bool inferenceSuccess =
            false;

        if (htpGate_ != nullptr) {

            inferenceSuccess =
                htpGate_->execute(
                    [&]() {

                        return
                            model_.infer();
                    },
                    htpWaitMs,
                    inferenceMs
                );
        }
        else {

            const Clock::time_point start =
                Clock::now();

            inferenceSuccess =
                model_.infer();

            const Clock::time_point end =
                Clock::now();

            inferenceMs =
                durationMs(
                    start,
                    end
                );
        }

        metric.queueMs =
            htpWaitMs;

        metric.inferenceMs =
            inferenceMs;

        if (!inferenceSuccess) {

            lastError_ =
                "FireSmoke inference failed: "
                +
                model_.lastError();

            return false;
        }

        // =================================================
        // POSTPROCESS
        //
        // defaults:
        // conf = 0.25
        // NMS  = 0.45
        // =================================================

        const Clock::time_point postprocessStart =
            Clock::now();

        std::vector<
            models::FireSmokeDetectionResult
        > detections;

        if (!model_.postprocess(
                detections
            )) {

            lastError_ =
                "FireSmoke postprocess failed: "
                +
                model_.lastError();

            return false;
        }

        const Clock::time_point postprocessEnd =
            Clock::now();

        metric.postprocessMs =
            durationMs(
                postprocessStart,
                postprocessEnd
            );

        metric.resultCount =
            detections.size();

        lastDetections_ =
            std::move(
                detections
            );

        updateTemporalState(
            lastDetections_
        );

        ++inferenceCount_;

        // =================================================
        // RENDER DETECTIONS + TEMPORAL STATE
        // =================================================

        const Clock::time_point renderStart =
            Clock::now();

        if (!lastDetections_.empty()) {

            cv::Mat rendered =
                model_.renderDetections(
                    renderFrame.image,
                    lastDetections_
                );

            if (rendered.empty()) {

                lastError_ =
                    "FireSmoke render returned empty image";

                return false;
            }

            renderFrame.image =
                std::move(
                    rendered
                );
        }

        renderStatusOverlay(
            renderFrame.image
        );

        const Clock::time_point renderEnd =
            Clock::now();

        metric.renderMs =
            durationMs(
                renderStart,
                renderEnd
            );

        metric.totalMs =
            durationMs(
                modelStart,
                renderEnd
            );

        modelMetrics.push_back(
            std::move(
                metric
            )
        );

        return true;
    }

    // =====================================================
    // Reuse latest boxes between FireSmoke inference frames.
    // =====================================================

    if (!lastDetections_.empty()) {

        cv::Mat rendered =
            model_.renderDetections(
                renderFrame.image,
                lastDetections_
            );

        if (rendered.empty()) {

            lastError_ =
                "Cached FireSmoke render returned empty image";

            return false;
        }

        renderFrame.image =
            std::move(
                rendered
            );
    }

    renderStatusOverlay(
        renderFrame.image
    );

    return true;
}

void FireSmokePipeline::updateTemporalState(
    const std::vector<
        models::FireSmokeDetectionResult
    >& detections
)
{
    bool fireDetected =
        false;

    bool smokeDetected =
        false;

    for (const auto& detection :
         detections) {

        // Class mapping intentionally follows
        // current FireSmoke model integration:
        //
        // 0 = smoke
        // 1 = fire

        if (detection.classId == 0) {

            smokeDetected =
                true;
        }

        else if (detection.classId == 1) {

            fireDetected =
                true;
        }
    }

    fireHistory_.push_back(
        fireDetected
    );

    smokeHistory_.push_back(
        smokeDetected
    );

    while (fireHistory_.size() >
           temporalWindow_) {

        fireHistory_.pop_front();
    }

    while (smokeHistory_.size() >
           temporalWindow_) {

        smokeHistory_.pop_front();
    }

    fireConfirmed_ =
        countTrue(
            fireHistory_
        )
        >=
        confirmationCount_;

    smokeConfirmed_ =
        countTrue(
            smokeHistory_
        )
        >=
        confirmationCount_;
}

void FireSmokePipeline::renderStatusOverlay(
    cv::Mat& image
) const
{
    if (image.empty()) {
        return;
    }

    int y =
        30;

    if (fireConfirmed_) {

        cv::putText(
            image,
            "FIRE CONFIRMED",
            cv::Point(
                20,
                y
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            cv::Scalar(
                0,
                0,
                255
            ),
            2,
            cv::LINE_AA
        );

        y +=
            35;
    }

    if (smokeConfirmed_) {

        cv::putText(
            image,
            "SMOKE CONFIRMED",
            cv::Point(
                20,
                y
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            cv::Scalar(
                160,
                160,
                160
            ),
            2,
            cv::LINE_AA
        );
    }
}

std::size_t FireSmokePipeline::countTrue(
    const std::deque<bool>& history
) noexcept
{
    return
        static_cast<std::size_t>(
            std::count(
                history.begin(),
                history.end(),
                true
            )
        );
}

uint64_t
FireSmokePipeline::inferenceCount() const noexcept
{
    return inferenceCount_;
}

bool
FireSmokePipeline::fireConfirmed() const noexcept
{
    return fireConfirmed_;
}

bool
FireSmokePipeline::smokeConfirmed() const noexcept
{
    return smokeConfirmed_;
}

std::size_t
FireSmokePipeline::firePositiveCount() const noexcept
{
    return
        countTrue(
            fireHistory_
        );
}

std::size_t
FireSmokePipeline::smokePositiveCount() const noexcept
{
    return
        countTrue(
            smokeHistory_
        );
}

const std::string&
FireSmokePipeline::lastError() const noexcept
{
    return lastError_;
}

double FireSmokePipeline::durationMs(
    Clock::time_point start,
    Clock::time_point end
) noexcept
{
    return
        std::chrono::duration<
            double,
            std::milli
        >(
            end - start
        ).count();
}

} // namespace pipeline