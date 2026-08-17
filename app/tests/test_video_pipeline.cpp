#include "metrics/metrics_logger.hpp"
#include "metrics/profiler.hpp"
#include "pipeline/video_pipeline.hpp"
#include "video/frame.hpp"
#include "video/video_file_sink.hpp"
#include "video/video_file_source.hpp"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

bool createInputVideo(
    const std::string& path
)
{
    constexpr int WIDTH =
        320;

    constexpr int HEIGHT =
        240;

    constexpr double FPS =
        10.0;

    constexpr int FRAME_COUNT =
        20;

    video::VideoFileSink sink(
        path
    );

    if (!sink.open(
            FPS,
            WIDTH,
            HEIGHT
        )) {

        std::cerr
            << "[ERROR] Synthetic VideoFileSink open failed:\n"
            << sink.lastError()
            << '\n';

        return false;
    }

    for (int i = 0;
         i < FRAME_COUNT;
         ++i) {

        const unsigned char value =
            static_cast<unsigned char>(
                i * 10
            );

        video::Frame frame;

        frame.frameId =
            static_cast<uint64_t>(
                i
            );

        frame.captureTimestampUs =
            static_cast<int64_t>(
                i
                *
                100000
            );

        frame.image =
            cv::Mat(
                HEIGHT,
                WIDTH,
                CV_8UC3,
                cv::Scalar(
                    value,
                    value,
                    value
                )
            );

        if (!sink.write(
                frame
            )) {

            std::cerr
                << "[ERROR] Synthetic write failed: "
                << sink.lastError()
                << '\n';

            sink.close();

            return false;
        }
    }

    sink.close();

    return true;
}

uint64_t countVideoFrames(
    const std::string& path
)
{
    cv::VideoCapture capture;

    if (!capture.open(
            path,
            cv::CAP_ANY
        )) {

        return 0;
    }

    uint64_t count =
        0;

    cv::Mat frame;

    while (capture.read(
        frame
    )) {

        if (frame.empty()) {
            break;
        }

        ++count;
    }

    capture.release();

    return
        count;
}

std::size_t countLines(
    const std::string& path
)
{
    std::ifstream file(
        path
    );

    if (!file) {

        return 0;
    }

    std::size_t lines =
        0;

    std::string line;

    while (std::getline(
        file,
        line
    )) {

        ++lines;
    }

    return
        lines;
}

} // namespace

int main()
{
    namespace fs =
        std::filesystem;

    constexpr uint64_t EXPECTED_FRAMES =
        20;

    const std::string inputPath =
        "/tmp/qcs6490_v2_input.mp4";

    const std::string outputPath =
        "/tmp/qcs6490_v2_output.mp4";

    const std::string metricsDirectory =
        "/tmp/qcs6490_v2_metrics";

    std::remove(
        inputPath.c_str()
    );

    std::remove(
        outputPath.c_str()
    );

    std::error_code error;

    fs::remove_all(
        metricsDirectory,
        error
    );

    // =====================================================
    // Synthetic input
    // =====================================================

    if (!createInputVideo(
            inputPath
        )) {

        std::cerr
            << "[ERROR] Cannot create synthetic input\n";

        return EXIT_FAILURE;
    }

    std::cout
        << "[PASS] synthetic input created\n";

    // =====================================================
    // Metrics infrastructure
    // =====================================================

    metrics::MetricsLogger logger;

    if (!logger.open(
            metricsDirectory
        )) {

        std::cerr
            << "[ERROR] Metrics logger: "
            << logger.lastError()
            << '\n';

        return EXIT_FAILURE;
    }

    metrics::PipelineProfiler profiler(
        1000.0
    );

    // =====================================================
    // Pipeline
    // =====================================================

    video::VideoFileSource source(
        inputPath
    );

    video::VideoFileSink sink(
        outputPath
    );

    pipeline::VideoPipeline pipeline(
        source,
        sink,
        &profiler,
        &logger
    );

    if (!pipeline.run()) {

        std::cerr
            << "[ERROR] pipeline.run(): "
            << pipeline.lastError()
            << '\n';

        return EXIT_FAILURE;
    }

    logger.close();

    std::cout
        << "[PASS] video pipeline executed\n";

    // =====================================================
    // Frame count
    // =====================================================

    const auto& stats =
        pipeline.stats();

    if (stats.framesRead !=
        EXPECTED_FRAMES) {

        std::cerr
            << "[ERROR] framesRead="
            << stats.framesRead
            << '\n';

        return EXIT_FAILURE;
    }

    if (stats.framesWritten !=
        EXPECTED_FRAMES) {

        std::cerr
            << "[ERROR] framesWritten="
            << stats.framesWritten
            << '\n';

        return EXIT_FAILURE;
    }

    std::cout
        << "[PASS] pipeline frame counts valid\n";

    const uint64_t outputFrames =
        countVideoFrames(
            outputPath
        );

    if (outputFrames !=
        EXPECTED_FRAMES) {

        std::cerr
            << "[ERROR] output video frames="
            << outputFrames
            << '\n';

        return EXIT_FAILURE;
    }

    std::cout
        << "[PASS] output video contains 20 frames\n";

    // =====================================================
    // Profiler
    // =====================================================

    const metrics::PipelineSummary summary =
        profiler.summary();

    if (summary.frameCount !=
        EXPECTED_FRAMES) {

        std::cerr
            << "[ERROR] profiler frameCount="
            << summary.frameCount
            << '\n';

        return EXIT_FAILURE;
    }

    if (summary.runtimeMs <= 0.0) {

        std::cerr
            << "[ERROR] profiler runtime invalid\n";

        return EXIT_FAILURE;
    }

    if (summary.effectiveFps <= 0.0) {

        std::cerr
            << "[ERROR] effective FPS invalid\n";

        return EXIT_FAILURE;
    }

    if (summary.averageTotalMs <= 0.0) {

        std::cerr
            << "[ERROR] average total latency invalid\n";

        return EXIT_FAILURE;
    }

    if (summary.p50TotalMs <= 0.0 ||
        summary.p95TotalMs <= 0.0 ||
        summary.p99TotalMs <= 0.0) {

        std::cerr
            << "[ERROR] percentile metrics invalid\n";

        return EXIT_FAILURE;
    }

    std::cout
        << "[PASS] profiler summary valid\n";

    // =====================================================
    // CSV files
    // =====================================================

    const std::string frameMetricsPath =
        metricsDirectory
        +
        "/frame_metrics.csv";

    const std::string modelMetricsPath =
        metricsDirectory
        +
        "/model_metrics.csv";

    const std::string systemMetricsPath =
        metricsDirectory
        +
        "/system_metrics.csv";

    if (!fs::exists(
            frameMetricsPath
        ) ||
        !fs::exists(
            modelMetricsPath
        ) ||
        !fs::exists(
            systemMetricsPath
        )) {

        std::cerr
            << "[ERROR] expected metric files are missing\n";

        return EXIT_FAILURE;
    }

    std::cout
        << "[PASS] all metric files created\n";

    // frame_metrics:
    //
    // header + 20 frames
    //
    const std::size_t frameMetricLines =
        countLines(
            frameMetricsPath
        );

    if (frameMetricLines !=
        EXPECTED_FRAMES + 1) {

        std::cerr
            << "[ERROR] frame_metrics lines="
            << frameMetricLines
            << ", expected "
            << EXPECTED_FRAMES + 1
            << '\n';

        return EXIT_FAILURE;
    }

    std::cout
        << "[PASS] frame_metrics contains 20 frame rows\n";

    // No AI yet:
    //
    // model_metrics should only contain its header.
    //
    const std::size_t modelMetricLines =
        countLines(
            modelMetricsPath
        );

    if (modelMetricLines != 1) {

        std::cerr
            << "[ERROR] model_metrics should contain "
            << "only header in V2\n";

        return EXIT_FAILURE;
    }

    std::cout
        << "[PASS] model_metrics V2 schema created\n";

    // Even a very short test must have at least one final
    // forced resource sample:
    //
    // header + >= 1 sample.
    //
    const std::size_t systemMetricLines =
        countLines(
            systemMetricsPath
        );

    if (systemMetricLines < 2) {

        std::cerr
            << "[ERROR] system_metrics has no samples\n";

        return EXIT_FAILURE;
    }

    std::cout
        << "[PASS] system metrics sample recorded\n";

    // =====================================================
    // Final diagnostic output
    // =====================================================

    std::cout
        << "[INFO] average read ms   : "
        << summary.averageReadMs
        << '\n';

    std::cout
        << "[INFO] average process ms: "
        << summary.averageProcessMs
        << '\n';

    std::cout
        << "[INFO] average write ms  : "
        << summary.averageWriteMs
        << '\n';

    std::cout
        << "[INFO] average total ms  : "
        << summary.averageTotalMs
        << '\n';

    std::cout
        << "[INFO] p50 total ms      : "
        << summary.p50TotalMs
        << '\n';

    std::cout
        << "[INFO] p95 total ms      : "
        << summary.p95TotalMs
        << '\n';

    std::cout
        << "[INFO] p99 total ms      : "
        << summary.p99TotalMs
        << '\n';

    std::cout
        << "[INFO] effective FPS     : "
        << summary.effectiveFps
        << '\n';

    std::cout
        << "[INFO] metrics directory : "
        << metricsDirectory
        << '\n';

    std::cout
        << "[PASS] V2 profiler + metrics test complete\n";

    return EXIT_SUCCESS;
}