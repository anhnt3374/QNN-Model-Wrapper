#pragma once

#include "metrics/metrics.hpp"

#include <chrono>
#include <cstdint>
#include <vector>

namespace metrics {

class PipelineProfiler {
public:
    using Clock =
        std::chrono::steady_clock;

    explicit PipelineProfiler(
        double systemSampleIntervalMs = 1000.0
    );

    void startRun();

    void finishRun();

    void reset();

    void recordFrame(
        const FrameMetrics& metrics
    );

    // Returns true when a system sample is produced.
    //
    // force = true:
    //     produce a sample regardless of interval.
    bool sampleSystemIfDue(
        SystemMetrics& output,
        bool force = false
    );

    PipelineSummary summary() const;

    uint64_t frameCount() const noexcept;

private:
    static double durationMs(
        Clock::time_point start,
        Clock::time_point end
    ) noexcept;

    static double average(
        const std::vector<double>& values
    );

    static double percentile(
        const std::vector<double>& values,
        double percentileValue
    );

    static double processCpuSeconds();

    static double processRssMb();

    static int processThreadCount();

    static double maxTemperatureC();

private:
    double systemSampleIntervalMs_ =
        1000.0;

    bool running_ =
        false;

    Clock::time_point runStart_{};

    Clock::time_point runEnd_{};

    Clock::time_point lastSystemSample_{};

    double lastCpuSeconds_ =
        0.0;

    uint64_t nextSystemSampleIndex_ =
        0;

    std::vector<double> readLatencies_;

    std::vector<double> processLatencies_;

    std::vector<double> writeLatencies_;

    std::vector<double> totalLatencies_;
};

} // namespace metrics