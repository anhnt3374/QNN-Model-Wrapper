#include "async_runtime/branch_worker.hpp"

#include <opencv2/core.hpp>

#include <chrono>
#include <utility>
#include <vector>

namespace async_runtime {

namespace {

using Clock =
    std::chrono::steady_clock;


double durationMs(
    Clock::time_point start,
    Clock::time_point end
)
{
    return
        std::chrono::duration<
            double,
            std::milli
        >(
            end - start
        ).count();
}

} // namespace


BranchWorker::BranchWorker(
    std::string name,
    pipeline::FrameProcessor& processor,
    OverlayResultCache& resultCache,
    AsyncMetricsLogger& metricsLogger,
    std::size_t queueCapacity
)
    : name_(
          std::move(
              name
          )
      ),
      processor_(
          processor
      ),
      resultCache_(
          resultCache
      ),
      metricsLogger_(
          metricsLogger
      ),
      queue_(
          queueCapacity
      )
{
}


BranchWorker::~BranchWorker()
{
    closeInput();

    join();
}


bool BranchWorker::configure(
    double sourceFps,
    int width,
    int height
)
{
    if (!processor_.configure(
            sourceFps,
            width,
            height
        )) {

        std::lock_guard<std::mutex> lock(
            errorMutex_
        );

        lastError_ =
            processor_.lastError();

        return false;
    }

    return true;
}


bool BranchWorker::start()
{
    if (running_) {

        return false;
    }

    failed_ =
        false;

    processedCount_ =
        0;

    running_ =
        true;

    thread_ =
        std::thread(
            &BranchWorker::run,
            this
        );

    return true;
}


bool BranchWorker::submit(
    AsyncFramePtr frame
)
{
    return
        queue_.push(
            std::move(
                frame
            )
        );
}


void BranchWorker::closeInput()
{
    queue_.close();
}


void BranchWorker::join()
{
    if (thread_.joinable()) {

        thread_.join();
    }
}


bool BranchWorker::failed() const noexcept
{
    return
        failed_.load();
}


const std::string&
BranchWorker::name() const noexcept
{
    return
        name_;
}


std::string BranchWorker::lastError() const
{
    std::lock_guard<std::mutex> lock(
        errorMutex_
    );

    return
        lastError_;
}


uint64_t
BranchWorker::processedCount() const noexcept
{
    return
        processedCount_.load();
}


uint64_t BranchWorker::droppedCount() const
{
    return
        queue_.droppedCount();
}


std::size_t
BranchWorker::maxQueueDepth() const
{
    return
        queue_.maxDepth();
}


void BranchWorker::run()
{
    while (true) {

        AsyncFramePtr asyncFrame;

        // =================================================
        // Always prefer freshest frame.
        //
        // If several accumulated while this worker was busy,
        // old frames are discarded here.
        // =================================================

        if (!queue_.waitPopLatest(
                asyncFrame
            )) {

            break;
        }

        if (!asyncFrame) {

            continue;
        }

        const Clock::time_point workerStart =
            Clock::now();

        const double workerQueueWaitMs =
            durationMs(
                asyncFrame->publishedAt,
                workerStart
            );

        const double frameAgeMs =
            workerQueueWaitMs;

        // =================================================
        // Worker-local render frame.
        //
        // Models read ORIGINAL sourceFrame.
        //
        // All rendering goes onto BLACK overlay.
        // =================================================

        video::Frame overlayFrame =
            asyncFrame->frame;

        overlayFrame.image =
            cv::Mat::zeros(
                asyncFrame->frame.image.size(),
                asyncFrame->frame.image.type()
            );

        std::vector<
            metrics::ModelMetrics
        > modelMetrics;

        const Clock::time_point processingStart =
            Clock::now();

        if (!processor_.process(
                asyncFrame->frame,
                overlayFrame,
                modelMetrics
            )) {

            {
                std::lock_guard<std::mutex> lock(
                    errorMutex_
                );

                lastError_ =
                    processor_.lastError();
            }

            failed_ =
                true;

            break;
        }

        const Clock::time_point processingEnd =
            Clock::now();

        const double processingMs =
            durationMs(
                processingStart,
                processingEnd
            );

        // =================================================
        // Store newest overlay.
        // =================================================

        OverlayResult result;

        result.sourceFrameId =
            asyncFrame->frame.frameId;

        result.sourceTimestampUs =
            asyncFrame->frame.captureTimestampUs;

        result.processingMs =
            processingMs;

        result.overlay =
            std::move(
                overlayFrame.image
            );

        resultCache_.update(
            std::move(
                result
            )
        );

        // =================================================
        // Worker/model metrics
        // =================================================

        for (const auto& metric :
             modelMetrics) {

            metricsLogger_.logModel(
                name_,
                metric,
                workerQueueWaitMs,
                frameAgeMs
            );
        }

        ++processedCount_;
    }

    running_ =
        false;
}

} // namespace async_runtime