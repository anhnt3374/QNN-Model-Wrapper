#include "pipeline/person_pipeline.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <utility>

namespace pipeline {

PersonPipeline::PersonPipeline(
    models::PersonDetectionModel& model,
    double targetInferenceFps
)
    : model_(
          model
      ),
      targetInferenceFps_(
          targetInferenceFps
      )
{
}

void PersonPipeline::setHtpExecutionGate(
    async_runtime::HtpExecutionGate* gate
) noexcept
{
    htpGate_ =
        gate;
}

bool PersonPipeline::configure(
    double sourceFps,
    int width,
    int height
)
{
    lastError_.clear();

    if (!model_.ready()) {

        lastError_ =
            "PersonDetectionModel is not ready";

        return false;
    }

    if (!std::isfinite(sourceFps) ||
        sourceFps <= 0.0) {

        lastError_ =
            "Invalid source FPS";

        return false;
    }

    if (!std::isfinite(targetInferenceFps_) ||
        targetInferenceFps_ <= 0.0) {

        lastError_ =
            "Invalid person inference FPS";

        return false;
    }

    if (width <= 0 ||
        height <= 0) {

        lastError_ =
            "Invalid video dimensions";

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
            "Invalid person inference period";

        return false;
    }

    scheduleInitialized_ =
        false;

    nextInferenceTimestampUs_ =
        0;

    inferenceCount_ =
        0;

    renderedFrameCount_ =
        0;

    lastDetections_.clear();

    return true;
}

bool PersonPipeline::shouldRunInference(
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

bool PersonPipeline::process(
    const video::Frame& sourceFrame,
    video::Frame& renderFrame,
    std::vector<metrics::ModelMetrics>& modelMetrics
)
{
    lastError_.clear();

    if (sourceFrame.image.empty()) {

        lastError_ =
            "PersonPipeline received empty source frame";

        return false;
    }

    if (renderFrame.image.empty()) {

        lastError_ =
            "PersonPipeline received empty render frame";

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
            "person";

        const Clock::time_point modelStart =
            Clock::now();

        // =================================================
        // PREPROCESS
        //
        // IMPORTANT:
        // Use original source frame, never rendered frame.
        // =================================================

        const Clock::time_point preprocessStart =
            Clock::now();

        if (!model_.preprocess(
                sourceFrame.image
            )) {

            lastError_ =
                "Person preprocess failed: "
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
        // INFERENCE
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

            const Clock::time_point inferenceStart =
                Clock::now();

            inferenceSuccess =
                model_.infer();

            const Clock::time_point inferenceEnd =
                Clock::now();

            inferenceMs =
                durationMs(
                    inferenceStart,
                    inferenceEnd
                );
        }

        metric.queueMs =
            htpWaitMs;

        metric.inferenceMs =
            inferenceMs;

        if (!inferenceSuccess) {

            lastError_ =
                "Person inference failed: "
                +
                model_.lastError();

            return false;
        }

        // =================================================
        // POSTPROCESS
        // =================================================

        const Clock::time_point postprocessStart =
            Clock::now();

        std::vector<
            models::PersonDetectionResult
        > detections;

        if (!model_.postprocess(
                detections
            )) {

            lastError_ =
                "Person postprocess failed: "
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

        ++inferenceCount_;

        // =================================================
        // RENDER
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
                    "Person render returned empty image";

                return false;
            }

            renderFrame.image =
                std::move(
                    rendered
                );

            ++renderedFrameCount_;
        }

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
    // Cached detection rendering
    // =====================================================

    if (!lastDetections_.empty()) {

        cv::Mat rendered =
            model_.renderDetections(
                renderFrame.image,
                lastDetections_
            );

        if (rendered.empty()) {

            lastError_ =
                "Cached person render returned empty image";

            return false;
        }

        renderFrame.image =
            std::move(
                rendered
            );

        ++renderedFrameCount_;
    }

    return true;
}

const std::string&
PersonPipeline::lastError() const noexcept
{
    return lastError_;
}

uint64_t
PersonPipeline::inferenceCount() const noexcept
{
    return inferenceCount_;
}

uint64_t
PersonPipeline::renderedFrameCount() const noexcept
{
    return renderedFrameCount_;
}

double
PersonPipeline::targetInferenceFps() const noexcept
{
    return targetInferenceFps_;
}

double PersonPipeline::durationMs(
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