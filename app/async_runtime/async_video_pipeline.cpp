#include "async_runtime/async_video_pipeline.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <thread>

namespace async_runtime {

AsyncVideoPipeline::AsyncVideoPipeline(
    video::VideoFileSource& source,
    video::VideoFileSink& sink,
    AsyncMetricsLogger& metricsLogger,
    std::size_t renderQueueCapacity
)
    : source_(
          source
      ),
      sink_(
          sink
      ),
      metricsLogger_(
          metricsLogger
      ),
      renderQueue_(
          renderQueueCapacity
      )
{
}


AsyncVideoPipeline::~AsyncVideoPipeline()
{
    renderQueue_.close();

    for (BranchWorker* worker :
         workers_) {

        if (worker != nullptr) {

            worker->closeInput();
        }
    }

    if (captureThread_.joinable()) {

        captureThread_.join();
    }

    for (BranchWorker* worker :
         workers_) {

        if (worker != nullptr) {

            worker->join();
        }
    }
}


void AsyncVideoPipeline::addWorker(
    BranchWorker& worker
)
{
    workers_.push_back(
        &worker
    );
}


void AsyncVideoPipeline::setPersonResultCache(
    OverlayResultCache* cache
) noexcept
{
    personResultCache_ =
        cache;
}


void AsyncVideoPipeline::setFireSmokeResultCache(
    OverlayResultCache* cache
) noexcept
{
    fireSmokeResultCache_ =
        cache;
}


void AsyncVideoPipeline::setAttendanceResultCache(
    OverlayResultCache* cache
) noexcept
{
    attendanceResultCache_ =
        cache;
}


bool AsyncVideoPipeline::run()
{
    lastError_.clear();

    capturedFrames_ =
        0;

    outputFrames_ =
        0;

    captureFailed_ =
        false;

    // =====================================================
    // Open input
    // =====================================================

    if (!source_.open()) {

        setError(
            "Cannot open input source: "
            +
            source_.lastError()
        );

        return false;
    }

    sourceFps_ =
        source_.fps();

    const int width =
        source_.width();

    const int height =
        source_.height();

    if (sourceFps_ <= 0.0 ||
        width <= 0 ||
        height <= 0) {

        setError(
            "Invalid source metadata"
        );

        source_.close();

        return false;
    }

    // =====================================================
    // Open output
    // =====================================================

    if (!sink_.open(
            sourceFps_,
            width,
            height
        )) {

        setError(
            "Cannot open output sink: "
            +
            sink_.lastError()
        );

        source_.close();

        return false;
    }

    // =====================================================
    // Configure all branch processors
    // =====================================================

    for (BranchWorker* worker :
         workers_) {

        if (worker == nullptr) {

            continue;
        }

        if (!worker->configure(
                sourceFps_,
                width,
                height
            )) {

            setError(
                worker->name()
                +
                ": "
                +
                worker->lastError()
            );

            sink_.close();

            source_.close();

            return false;
        }
    }

    // =====================================================
    // Start AI workers
    // =====================================================

    for (BranchWorker* worker :
         workers_) {

        if (worker == nullptr) {

            continue;
        }

        if (!worker->start()) {

            setError(
                "Cannot start worker: "
                +
                worker->name()
            );

            return false;
        }
    }

    // =====================================================
    // Start realtime-paced capture thread
    // =====================================================

    captureThread_ =
        std::thread(
            &AsyncVideoPipeline::captureLoop,
            this,
            sourceFps_
        );

    // =====================================================
    // Renderer runs on calling/main thread.
    // =====================================================

    AsyncFramePtr asyncFrame;

    while (renderQueue_.waitPopLatest(
        asyncFrame
    )) {

        if (!asyncFrame) {

            continue;
        }

        for (BranchWorker* worker :
             workers_) {

            if (worker != nullptr &&
                worker->failed()) {

                setError(
                    worker->name()
                    +
                    ": "
                    +
                    worker->lastError()
                );

                break;
            }
        }

        if (!lastError_.empty()) {

            break;
        }

        const Clock::time_point renderStart =
            Clock::now();

        video::Frame outputFrame =
            asyncFrame->frame;

        // Deep copy:
        //
        // never draw directly on shared source Mat.
        outputFrame.image =
            asyncFrame->frame.image.clone();

        double personAgeMs =
            -1.0;

        double fireAgeMs =
            -1.0;

        double attendanceAgeMs =
            -1.0;

        composeOverlay(
            outputFrame.image,
            personResultCache_,
            outputFrame.captureTimestampUs,
            personAgeMs
        );

        composeOverlay(
            outputFrame.image,
            fireSmokeResultCache_,
            outputFrame.captureTimestampUs,
            fireAgeMs
        );

        composeOverlay(
            outputFrame.image,
            attendanceResultCache_,
            outputFrame.captureTimestampUs,
            attendanceAgeMs
        );

        const Clock::time_point writeStart =
            Clock::now();

        if (!sink_.write(
                outputFrame
            )) {

            setError(
                "Output write failed: "
                +
                sink_.lastError()
            );

            break;
        }

        const Clock::time_point writeEnd =
            Clock::now();

        ++outputFrames_;

        AsyncFrameMetrics frameMetric;

        frameMetric.frameId =
            outputFrame.frameId;

        frameMetric.captureTimestampUs =
            outputFrame.captureTimestampUs;

        frameMetric.readMs =
            asyncFrame->readMs;

        frameMetric.writeMs =
            durationMs(
                writeStart,
                writeEnd
            );

        frameMetric.frameAgeMs =
            durationMs(
                asyncFrame->publishedAt,
                writeEnd
            );

        frameMetric.renderQueueDepth =
            renderQueue_.size();

        frameMetric.renderDroppedFrames =
            renderQueue_.droppedCount();

        frameMetric.personResultAgeMs =
            personAgeMs;

        frameMetric.fireSmokeResultAgeMs =
            fireAgeMs;

        frameMetric.attendanceResultAgeMs =
            attendanceAgeMs;

        metricsLogger_.logFrame(
            frameMetric
        );

        (void)renderStart;
    }

    // =====================================================
    // Shutdown
    // =====================================================

    if (captureThread_.joinable()) {

        captureThread_.join();
    }

    for (BranchWorker* worker :
         workers_) {

        if (worker != nullptr) {

            worker->closeInput();

            worker->join();
        }
    }

    sink_.close();

    source_.close();

    if (captureFailed_) {

        return false;
    }

    for (BranchWorker* worker :
         workers_) {

        if (worker != nullptr &&
            worker->failed()) {

            setError(
                worker->name()
                +
                ": "
                +
                worker->lastError()
            );

            return false;
        }
    }

    return
        lastError_.empty();
}


void AsyncVideoPipeline::captureLoop(
    double sourceFps
)
{
    const double framePeriodUs =
        1'000'000.0
        /
        sourceFps;

    bool firstFrame =
        true;

    int64_t firstTimestampUs =
        0;

    uint64_t firstFrameId =
        0;

    Clock::time_point realtimeStart;

    uint64_t sequence =
        0;

    while (true) {

        video::Frame frame;

        const Clock::time_point readStart =
            Clock::now();

        if (!source_.read(
                frame
            )) {

            // For current VideoFileSource:
            // false at EOF is normal.
            break;
        }

        const Clock::time_point readEnd =
            Clock::now();

        const double readMs =
            durationMs(
                readStart,
                readEnd
            );

        if (firstFrame) {

            firstFrame =
                false;

            firstTimestampUs =
                frame.captureTimestampUs;

            firstFrameId =
                frame.frameId;

            realtimeStart =
                Clock::now();
        }

        // =================================================
        // Realtime pacing.
        //
        // Prefer source timestamps when valid.
        // Fall back to frameId/FPS.
        // =================================================

        int64_t relativeUs =
            frame.captureTimestampUs
            -
            firstTimestampUs;

        if (relativeUs < 0) {

            relativeUs =
                static_cast<int64_t>(
                    std::llround(
                        static_cast<double>(
                            frame.frameId
                            -
                            firstFrameId
                        )
                        *
                        framePeriodUs
                    )
                );
        }

        const Clock::time_point target =
            realtimeStart
            +
            std::chrono::microseconds(
                relativeUs
            );

        const Clock::time_point now =
            Clock::now();

        if (target >
            now) {

            std::this_thread::sleep_until(
                target
            );
        }

        auto asyncFrame =
            std::make_shared<
                AsyncFrame
            >();

        asyncFrame->frame =
            std::move(
                frame
            );

        asyncFrame->publishedAt =
            Clock::now();

        asyncFrame->readMs =
            readMs;

        asyncFrame->sequence =
            sequence++;

        AsyncFramePtr sharedFrame =
            asyncFrame;

        // =================================================
        // Publish same immutable frame to every branch.
        // =================================================

        for (BranchWorker* worker :
             workers_) {

            if (worker != nullptr) {

                worker->submit(
                    sharedFrame
                );
            }
        }

        renderQueue_.push(
            sharedFrame
        );

        ++capturedFrames_;
    }

    // =====================================================
    // End-of-stream
    // =====================================================

    for (BranchWorker* worker :
         workers_) {

        if (worker != nullptr) {

            worker->closeInput();
        }
    }

    renderQueue_.close();
}


bool AsyncVideoPipeline::composeOverlay(
    cv::Mat& output,
    const OverlayResultCache* cache,
    int64_t currentTimestampUs,
    double& ageMs
)
{
    ageMs =
        -1.0;

    if (cache == nullptr) {

        return false;
    }

    OverlayResult result;

    if (!cache->latest(
            result
        )) {

        return false;
    }

    if (result.overlay.empty()) {

        return false;
    }

    // Do not draw a result from a future source frame.
    if (result.sourceTimestampUs >
        currentTimestampUs) {

        return false;
    }

    ageMs =
        static_cast<double>(
            currentTimestampUs
            -
            result.sourceTimestampUs
        )
        /
        1000.0;

    if (result.overlay.size() !=
            output.size() ||
        result.overlay.type() !=
            output.type()) {

        return false;
    }

    // =====================================================
    // Overlay frame has black background.
    //
    // Convert it to mask:
    //
    // non-black pixel => copy to output.
    // =====================================================

    cv::Mat gray;

    if (result.overlay.channels() == 3) {

        cv::cvtColor(
            result.overlay,
            gray,
            cv::COLOR_BGR2GRAY
        );
    }
    else {

        gray =
            result.overlay;
    }

    cv::Mat mask;

    cv::threshold(
        gray,
        mask,
        0,
        255,
        cv::THRESH_BINARY
    );

    result.overlay.copyTo(
        output,
        mask
    );

    return true;
}


double AsyncVideoPipeline::durationMs(
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


void AsyncVideoPipeline::setError(
    const std::string& error
)
{
    std::lock_guard<std::mutex> lock(
        errorMutex_
    );

    if (lastError_.empty()) {

        lastError_ =
            error;
    }
}


const std::string&
AsyncVideoPipeline::lastError() const noexcept
{
    return
        lastError_;
}


uint64_t
AsyncVideoPipeline::capturedFrames() const noexcept
{
    return
        capturedFrames_.load();
}


uint64_t
AsyncVideoPipeline::outputFrames() const noexcept
{
    return
        outputFrames_.load();
}


uint64_t
AsyncVideoPipeline::droppedRenderFrames() const
{
    return
        renderQueue_.droppedCount();
}

} // namespace async_runtime