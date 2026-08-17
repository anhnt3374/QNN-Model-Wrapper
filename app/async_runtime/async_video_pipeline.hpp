#pragma once

#include "async_runtime/async_frame.hpp"
#include "async_runtime/async_metrics_logger.hpp"
#include "async_runtime/branch_worker.hpp"
#include "async_runtime/latest_frame_queue.hpp"
#include "async_runtime/overlay_result_cache.hpp"

#include "video/video_file_sink.hpp"
#include "video/video_file_source.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace async_runtime {

class AsyncVideoPipeline {
public:
    AsyncVideoPipeline(
        video::VideoFileSource& source,
        video::VideoFileSink& sink,
        AsyncMetricsLogger& metricsLogger,
        std::size_t renderQueueCapacity = 2
    );

    ~AsyncVideoPipeline();

    AsyncVideoPipeline(
        const AsyncVideoPipeline&
    ) = delete;

    AsyncVideoPipeline& operator=(
        const AsyncVideoPipeline&
    ) = delete;

    AsyncVideoPipeline(
        AsyncVideoPipeline&&
    ) = delete;

    AsyncVideoPipeline& operator=(
        AsyncVideoPipeline&&
    ) = delete;

    // =====================================================
    // Workers
    // =====================================================

    void addWorker(
        BranchWorker& worker
    );

    // =====================================================
    // Latest result caches
    // =====================================================

    void setPersonResultCache(
        OverlayResultCache* cache
    ) noexcept;

    void setFireSmokeResultCache(
        OverlayResultCache* cache
    ) noexcept;

    void setAttendanceResultCache(
        OverlayResultCache* cache
    ) noexcept;

    // =====================================================
    // Run
    // =====================================================

    bool run();

    // =====================================================
    // Status
    // =====================================================

    const std::string&
    lastError() const noexcept;

    uint64_t
    capturedFrames() const noexcept;

    uint64_t
    outputFrames() const noexcept;

    uint64_t
    droppedRenderFrames() const;

private:
    using Clock =
        std::chrono::steady_clock;

    // =====================================================
    // Capture
    //
    // Video file is paced according to source timestamps /
    // source FPS so this behaves like a realtime source.
    // =====================================================

    void captureLoop(
        double sourceFps
    );

    // =====================================================
    // Rendering
    //
    // Apply a black-background overlay result onto the
    // current original frame.
    // =====================================================

    bool composeOverlay(
        cv::Mat& output,
        const OverlayResultCache* cache,
        int64_t currentTimestampUs,
        double& ageMs
    );

    // =====================================================
    // Helpers
    // =====================================================

    static double durationMs(
        Clock::time_point start,
        Clock::time_point end
    ) noexcept;

    void setError(
        const std::string& error
    );

private:
    // =====================================================
    // Video I/O
    // =====================================================

    video::VideoFileSource&
        source_;

    video::VideoFileSink&
        sink_;

    // =====================================================
    // Metrics
    // =====================================================

    AsyncMetricsLogger&
        metricsLogger_;

    // =====================================================
    // AI workers
    //
    // Non-owning pointers.
    // Worker lifetime must exceed AsyncVideoPipeline::run().
    // =====================================================

    std::vector<
        BranchWorker*
    > workers_;

    // =====================================================
    // Render queue
    //
    // Capacity normally = 2.
    // Old frames are dropped when renderer is behind.
    // =====================================================

    LatestFrameQueue<
        AsyncFramePtr
    > renderQueue_;

    // =====================================================
    // Latest branch results
    //
    // Non-owning.
    // =====================================================

    OverlayResultCache*
        personResultCache_ =
            nullptr;

    OverlayResultCache*
        fireSmokeResultCache_ =
            nullptr;

    OverlayResultCache*
        attendanceResultCache_ =
            nullptr;

    // =====================================================
    // Capture thread
    // =====================================================

    std::thread
        captureThread_;

    std::atomic<bool>
        captureFailed_{
            false
        };

    // =====================================================
    // Counters
    // =====================================================

    std::atomic<uint64_t>
        capturedFrames_{
            0
        };

    std::atomic<uint64_t>
        outputFrames_{
            0
        };

    // =====================================================
    // Error state
    // =====================================================

    mutable std::mutex
        errorMutex_;

    std::string
        lastError_;

    // =====================================================
    // Source metadata
    // =====================================================

    double
        sourceFps_ =
            0.0;
};

} // namespace async_runtime