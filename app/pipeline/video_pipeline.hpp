#pragma once

#include "metrics/metrics_logger.hpp"
#include "metrics/profiler.hpp"
#include "video/frame_sink.hpp"
#include "video/frame_source.hpp"

#include <cstdint>
#include <string>

namespace pipeline {

struct VideoPipelineStats {
    uint64_t framesRead = 0;

    uint64_t framesWritten = 0;
};

class VideoPipeline {
public:
    VideoPipeline(
        video::FrameSource& source,
        video::FrameSink& sink,
        metrics::PipelineProfiler* profiler = nullptr,
        metrics::MetricsLogger* metricsLogger = nullptr
    ) noexcept;

    VideoPipeline(
        const VideoPipeline&
    ) = delete;

    VideoPipeline& operator=(
        const VideoPipeline&
    ) = delete;

    bool run();

    const VideoPipelineStats&
    stats() const noexcept;

    const std::string&
    lastError() const noexcept;

private:
    using Clock =
        metrics::PipelineProfiler::Clock;

    static double durationMs(
        Clock::time_point start,
        Clock::time_point end
    ) noexcept;

private:
    video::FrameSource& source_;

    video::FrameSink& sink_;

    metrics::PipelineProfiler* profiler_ =
        nullptr;

    metrics::MetricsLogger* metricsLogger_ =
        nullptr;

    VideoPipelineStats stats_;

    std::string lastError_;
};

} // namespace pipeline