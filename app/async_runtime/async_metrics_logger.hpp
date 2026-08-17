#pragma once

#include "metrics/metrics.hpp"

#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>

namespace async_runtime {

struct AsyncFrameMetrics {
    uint64_t frameId =
        0;

    int64_t captureTimestampUs =
        0;

    double readMs =
        0.0;

    double writeMs =
        0.0;

    double frameAgeMs =
        0.0;

    std::size_t renderQueueDepth =
        0;

    uint64_t renderDroppedFrames =
        0;

    double personResultAgeMs =
        -1.0;

    double fireSmokeResultAgeMs =
        -1.0;

    double attendanceResultAgeMs =
        -1.0;
};


class AsyncMetricsLogger {
public:
    AsyncMetricsLogger() = default;

    ~AsyncMetricsLogger();

    bool open(
        const std::string& directory
    );

    void close();

    bool logModel(
        const std::string& branch,
        const metrics::ModelMetrics& metric,
        double workerQueueWaitMs,
        double frameAgeMs
    );

    bool logFrame(
        const AsyncFrameMetrics& metric
    );

    const std::string&
    lastError() const noexcept;

private:
    std::mutex mutex_;

    std::ofstream modelFile_;

    std::ofstream frameFile_;

    std::string lastError_;
};

} // namespace async_runtime