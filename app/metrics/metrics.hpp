#pragma once

#include <cstdint>
#include <string>

namespace metrics {

// =========================================================
// One row per processed frame.
//
// V2 uses:
//   readMs
//   processMs
//   writeMs
//   totalMs
//
// Fields for AI/queue can be added later without changing
// FrameSource / FrameSink.
// =========================================================

struct FrameMetrics {
    uint64_t frameId = 0;

    int64_t captureTimestampUs = 0;

    double readMs = 0.0;

    double processMs = 0.0;

    double writeMs = 0.0;

    double totalMs = 0.0;

    double instantaneousFps = 0.0;
};

// =========================================================
// Future model-level metric.
//
// The CSV is already created in V2 even though no model
// writes rows yet.
//
// Starting from V3:
//   PersonDetectionModel
//
// Later:
//   SCRFD
//   EdgeFace
//   FireSmoke
// =========================================================

struct ModelMetrics {
    uint64_t frameId = 0;

    std::string model;

    double preprocessMs = 0.0;

    double queueMs = 0.0;

    double inferenceMs = 0.0;

    double postprocessMs = 0.0;

    double totalMs = 0.0;

    uint64_t resultCount = 0;
};

// =========================================================
// Process/system resource sample.
// =========================================================

struct SystemMetrics {
    uint64_t sampleIndex = 0;

    // Time since pipeline start.
    double elapsedMs = 0.0;

    // Process CPU usage between two samples.
    //
    // This may exceed 100% when multiple CPU cores are used.
    double cpuPercent = 0.0;

    // Resident Set Size of the process.
    double rssMb = 0.0;

    // Maximum readable thermal zone temperature.
    //
    // -1 means unavailable.
    double temperatureC = -1.0;

    // Linux process thread count.
    int threadCount = 0;
};

// =========================================================
// Final runtime summary.
// =========================================================

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