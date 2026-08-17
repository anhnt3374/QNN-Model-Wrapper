#include "metrics/metrics_logger.hpp"
#include "metrics/profiler.hpp"
#include "pipeline/video_pipeline.hpp"
#include "video/video_file_sink.hpp"
#include "video/video_file_source.hpp"

#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

int main(
    int argc,
    char** argv
)
{
    if (argc < 3 ||
        argc > 5) {

        std::cerr
            << "Usage:\n"
            << "  "
            << argv[0]
            << " <input.mp4>"
            << " <output.mp4>"
            << " [metrics_dir]"
            << " [bitrate_kbps]\n\n"
            << "Examples:\n"
            << "  "
            << argv[0]
            << " input.mp4 output.mp4\n\n"
            << "  "
            << argv[0]
            << " input.mp4 output.mp4 logs\n\n"
            << "  "
            << argv[0]
            << " input.mp4 output.mp4 logs 4000\n";

        return EXIT_FAILURE;
    }

    const std::string inputPath =
        argv[1];

    const std::string outputPath =
        argv[2];

    const std::string metricsDirectory =
        argc >= 4
            ? argv[3]
            : "logs";

    int bitrateKbps =
        4000;

    if (argc == 5) {

        try {

            bitrateKbps =
                std::stoi(
                    argv[4]
                );
        }
        catch (const std::exception&) {

            std::cerr
                << "[ERROR] Invalid bitrate: "
                << argv[4]
                << '\n';

            return EXIT_FAILURE;
        }

        if (bitrateKbps <= 0) {

            std::cerr
                << "[ERROR] bitrate_kbps must be > 0\n";

            return EXIT_FAILURE;
        }
    }

    // =====================================================
    // Metrics
    // =====================================================

    metrics::MetricsLogger metricsLogger;

    if (!metricsLogger.open(
            metricsDirectory
        )) {

        std::cerr
            << "[ERROR] Cannot initialize metrics logger: "
            << metricsLogger.lastError()
            << '\n';

        return EXIT_FAILURE;
    }

    metrics::PipelineProfiler profiler(
        1000.0
    );

    // =====================================================
    // Video source
    // =====================================================

    video::VideoFileSource source(
        inputPath
    );

    // =====================================================
    // QCS6490 H.264 hardware sink
    // =====================================================

    video::VideoFileSink sink(
        outputPath,
        bitrateKbps
    );

    // =====================================================
    // Pipeline
    // =====================================================

    pipeline::VideoPipeline videoPipeline(
        source,
        sink,
        &profiler,
        &metricsLogger
    );

    std::cout
        << "========================================\n"
        << "QCS6490 VIDEO PIPELINE - V2\n"
        << "========================================\n"
        << "Input       : "
        << inputPath
        << '\n'
        << "Output      : "
        << outputPath
        << '\n'
        << "Metrics     : "
        << metricsDirectory
        << '\n'
        << "Encoder     : v4l2h264enc\n"
        << "Container   : MP4\n"
        << "========================================\n";

    if (!videoPipeline.run()) {

        std::cerr
            << "[ERROR] Pipeline failed: "
            << videoPipeline.lastError()
            << '\n';

        metricsLogger.close();

        return EXIT_FAILURE;
    }

    metricsLogger.close();

    const auto& stats =
        videoPipeline.stats();

    const metrics::PipelineSummary summary =
        profiler.summary();

    std::cout
        << std::fixed
        << std::setprecision(3);

    std::cout
        << "\n========================================\n"
        << "VIDEO PIPELINE SUMMARY\n"
        << "========================================\n";

    std::cout
        << "Frames read       : "
        << stats.framesRead
        << '\n';

    std::cout
        << "Frames written    : "
        << stats.framesWritten
        << '\n';

    std::cout
        << "Runtime           : "
        << summary.runtimeMs
        << " ms\n";

    std::cout
        << "Effective FPS     : "
        << summary.effectiveFps
        << '\n';

    std::cout
        << '\n'
        << "Average latency\n"
        << "----------------------------------------\n";

    std::cout
        << "Read              : "
        << summary.averageReadMs
        << " ms\n";

    std::cout
        << "Process           : "
        << summary.averageProcessMs
        << " ms\n";

    std::cout
        << "Write             : "
        << summary.averageWriteMs
        << " ms\n";

    std::cout
        << "Total/frame       : "
        << summary.averageTotalMs
        << " ms\n";

    std::cout
        << '\n'
        << "Frame total latency percentiles\n"
        << "----------------------------------------\n";

    std::cout
        << "P50               : "
        << summary.p50TotalMs
        << " ms\n";

    std::cout
        << "P95               : "
        << summary.p95TotalMs
        << " ms\n";

    std::cout
        << "P99               : "
        << summary.p99TotalMs
        << " ms\n";

    std::cout
        << "MAX               : "
        << summary.maxTotalMs
        << " ms\n";

    std::cout
        << '\n'
        << "Metrics files\n"
        << "----------------------------------------\n"
        << metricsDirectory
        << "/frame_metrics.csv\n"
        << metricsDirectory
        << "/model_metrics.csv\n"
        << metricsDirectory
        << "/system_metrics.csv\n"
        << "========================================\n"
        << std::defaultfloat;

    std::cout
        << "[PASS] V2 profiler + metrics complete\n";

    return EXIT_SUCCESS;
}