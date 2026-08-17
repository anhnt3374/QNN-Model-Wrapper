#pragma once

#include <cstdint>
#include <string>

namespace metrics {

struct FrameMetrics {
    uint64_t frameId = 0;

    int64_t captureTimestampUs = 0;

    double readMs = 0.0;

    double processMs = 0.0;

    double writeMs = 0.0;

    double totalMs = 0.0;

    double instantaneousFps = 0.0;
};

struct ModelMetrics {
    uint64_t frameId = 0;

    std::string model;

    double preprocessMs = 0.0;

    double queueMs = 0.0;

    double inferenceMs = 0.0;

    double postprocessMs = 0.0;

    // Overlay / drawing cost.
    double renderMs = 0.0;

    double totalMs = 0.0;

    uint64_t resultCount = 0;
};

struct SystemMetrics {
    uint64_t sampleIndex = 0;

    double elapsedMs = 0.0;

    double cpuPercent = 0.0;

    double rssMb = 0.0;

    double temperatureC = -1.0;

    int threadCount = 0;
};

struct PipelineSummary {
    uint64_t frameCount = 0;

    double runtimeMs = 0.0;

    double effectiveFps = 0.0;

    double averageReadMs = 0.0;

    double averageProcessMs = 0.0;

    double averageWriteMs = 0.0;

    double averageTotalMs = 0.0;

    double p50TotalMs = 0.0;

    double p95TotalMs = 0.0;

    double p99TotalMs = 0.0;

    double maxTotalMs = 0.0;
};

} // namespace metrics