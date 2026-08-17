#pragma once

#include "async_runtime/async_frame.hpp"
#include "async_runtime/async_metrics_logger.hpp"
#include "async_runtime/latest_frame_queue.hpp"
#include "async_runtime/overlay_result_cache.hpp"

#include "pipeline/frame_processor.hpp"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

namespace async_runtime {

class BranchWorker {
public:
    BranchWorker(
        std::string name,
        pipeline::FrameProcessor& processor,
        OverlayResultCache& resultCache,
        AsyncMetricsLogger& metricsLogger,
        std::size_t queueCapacity = 2
    );

    ~BranchWorker();

    BranchWorker(
        const BranchWorker&
    ) = delete;

    BranchWorker& operator=(
        const BranchWorker&
    ) = delete;

    bool configure(
        double sourceFps,
        int width,
        int height
    );

    bool start();

    bool submit(
        AsyncFramePtr frame
    );

    void closeInput();

    void join();

    bool failed() const noexcept;

    const std::string&
    name() const noexcept;

    std::string lastError() const;

    uint64_t processedCount() const noexcept;

    uint64_t droppedCount() const;

    std::size_t maxQueueDepth() const;

private:
    void run();

private:
    std::string name_;

    pipeline::FrameProcessor&
        processor_;

    OverlayResultCache&
        resultCache_;

    AsyncMetricsLogger&
        metricsLogger_;

    LatestFrameQueue<
        AsyncFramePtr
    > queue_;

    std::thread thread_;

    std::atomic<bool> running_{
        false
    };

    std::atomic<bool> failed_{
        false
    };

    std::atomic<uint64_t> processedCount_{
        0
    };

    mutable std::mutex errorMutex_;

    std::string lastError_;
};

} // namespace async_runtime