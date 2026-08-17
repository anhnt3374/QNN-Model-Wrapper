#include "config/fire_smoke_config.hpp"

#include "inference/qnn_backend.hpp"

#include "metrics/metrics_logger.hpp"
#include "metrics/profiler.hpp"

#include "models/fire_smoke_detection_model.hpp"
#include "models/person_detection_model.hpp"

#include "pipeline/composite_frame_processor.hpp"
#include "pipeline/fire_smoke_pipeline.hpp"
#include "pipeline/person_pipeline.hpp"
#include "pipeline/video_pipeline.hpp"

#include "video/video_file_sink.hpp"
#include "video/video_file_source.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

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

    std::size_t count =
        0;

    std::string line;

    while (std::getline(
        file,
        line
    )) {

        ++count;
    }

    return count;
}

} // namespace

int main(
    int argc,
    char** argv
)
{
    if (argc != 4) {

        std::cerr
            << "Usage:\n"
            << "  "
            << argv[0]
            << " <input.mp4>"
            << " <output.mp4>"
            << " <metrics_dir>\n";

        return EXIT_FAILURE;
    }

    const char* backendPath =
        std::getenv(
            "QNN_BACKEND_PATH"
        );

    const char* personModelPath =
        std::getenv(
            "QNN_PERSON_MODEL_PATH"
        );

    const char* fireSmokeModelPath =
        std::getenv(
            "QNN_FIRE_SMOKE_MODEL_PATH"
        );

    if (backendPath == nullptr ||
        personModelPath == nullptr ||
        fireSmokeModelPath == nullptr) {

        std::cerr
            << "[ERROR] Required QNN environment missing\n";

        return EXIT_FAILURE;
    }

    models::FireSmokeAnchors fireSmokeAnchors;

    std::string anchorError;

    if (!config::loadFireSmokeAnchorsFromEnvironment(
            fireSmokeAnchors,
            anchorError
        )) {

        std::cerr
            << "[ERROR] anchors: "
            << anchorError
            << '\n';

        return EXIT_FAILURE;
    }

    // =====================================================
    // Backend
    // =====================================================

    inference::QnnBackend backend;

    if (!backend.loadLibrary(
            backendPath
        ) ||
        !backend.loadProviders() ||
        !backend.selectInterface() ||
        !backend.createBackend() ||
        !backend.createDevice()) {

        std::cerr
            << "[ERROR] backend: "
            << backend.lastError()
            << '\n';

        return EXIT_FAILURE;
    }

    std::cout
        << "[PASS] QNN backend ready\n";

    // =====================================================
    // Person
    // =====================================================

    models::PersonDetectionModel personModel(
        backend
    );

    if (!personModel.initialize(
            personModelPath
        )) {

        std::cerr
            << "[ERROR] person: "
            << personModel.lastError()
            << '\n';

        return EXIT_FAILURE;
    }

    std::cout
        << "[PASS] Person model ready\n";

    // =====================================================
    // FireSmoke
    // =====================================================

    models::FireSmokeDetectionModel fireSmokeModel(
        backend
    );

    if (!fireSmokeModel.initialize(
            fireSmokeModelPath
        )) {

        std::cerr
            << "[ERROR] firesmoke: "
            << fireSmokeModel.lastError()
            << '\n';

        return EXIT_FAILURE;
    }

    if (!fireSmokeModel.setAnchors(
            fireSmokeAnchors
        )) {

        std::cerr
            << "[ERROR] FireSmoke anchors: "
            << fireSmokeModel.lastError()
            << '\n';

        return EXIT_FAILURE;
    }

    std::cout
        << "[PASS] FireSmoke model ready\n";

    // =====================================================
    // Processor
    // =====================================================

    pipeline::PersonPipeline personPipeline(
        personModel,
        10.0
    );

    pipeline::FireSmokePipeline fireSmokePipeline(
        fireSmokeModel,
        5.0,
        5,
        3
    );

    pipeline::CompositeFrameProcessor composite;

    composite.addProcessor(
        personPipeline
    );

    composite.addProcessor(
        fireSmokePipeline
    );

    if (composite.processorCount() != 2) {

        std::cerr
            << "[ERROR] composite processor count invalid\n";

        return EXIT_FAILURE;
    }

    // =====================================================
    // Metrics
    // =====================================================

    std::filesystem::remove_all(
        argv[3]
    );

    metrics::MetricsLogger logger;

    if (!logger.open(
            argv[3]
        )) {

        std::cerr
            << "[ERROR] logger: "
            << logger.lastError()
            << '\n';

        return EXIT_FAILURE;
    }

    metrics::PipelineProfiler profiler;

    // =====================================================
    // Video
    // =====================================================

    video::VideoFileSource source(
        argv[1]
    );

    video::VideoFileSink sink(
        argv[2]
    );

    pipeline::VideoPipeline pipeline(
        source,
        sink,
        &profiler,
        &logger,
        &composite
    );

    if (!pipeline.run()) {

        std::cerr
            << "[ERROR] pipeline: "
            << pipeline.lastError()
            << '\n';

        return EXIT_FAILURE;
    }

    logger.close();

    // =====================================================
    // Validation
    // =====================================================

    if (pipeline.stats().framesRead == 0) {

        std::cerr
            << "[ERROR] zero frames\n";

        return EXIT_FAILURE;
    }

    if (pipeline.stats().framesRead !=
        pipeline.stats().framesWritten) {

        std::cerr
            << "[ERROR] read/write count mismatch\n";

        return EXIT_FAILURE;
    }

    if (personPipeline.inferenceCount() == 0) {

        std::cerr
            << "[ERROR] no Person inference\n";

        return EXIT_FAILURE;
    }

    if (fireSmokePipeline.inferenceCount() == 0) {

        std::cerr
            << "[ERROR] no FireSmoke inference\n";

        return EXIT_FAILURE;
    }

    if (fireSmokePipeline.inferenceCount() >=
        personPipeline.inferenceCount()) {

        std::cerr
            << "[ERROR] expected FireSmoke FPS "
            << "to be lower than Person FPS\n";

        return EXIT_FAILURE;
    }

    const std::string modelMetricsPath =
        std::string(
            argv[3]
        )
        +
        "/model_metrics.csv";

    const std::size_t modelLines =
        countLines(
            modelMetricsPath
        );

    const uint64_t expectedMetricRows =
        personPipeline.inferenceCount()
        +
        fireSmokePipeline.inferenceCount();

    if (modelLines !=
        expectedMetricRows + 1) {

        std::cerr
            << "[ERROR] model metric rows mismatch: "
            << modelLines
            << " vs expected "
            << expectedMetricRows + 1
            << '\n';

        return EXIT_FAILURE;
    }

    const metrics::PipelineSummary summary =
        profiler.summary();

    if (summary.frameCount !=
        pipeline.stats().framesRead) {

        std::cerr
            << "[ERROR] profiler frame count mismatch\n";

        return EXIT_FAILURE;
    }

    // =====================================================
    // Output
    // =====================================================

    std::cout
        << "[PASS] frames read/written valid\n";

    std::cout
        << "[PASS] both model schedulers executed\n";

    std::cout
        << "[INFO] frames: "
        << pipeline.stats().framesRead
        << '\n';

    std::cout
        << "[INFO] person inference count: "
        << personPipeline.inferenceCount()
        << '\n';

    std::cout
        << "[INFO] firesmoke inference count: "
        << fireSmokePipeline.inferenceCount()
        << '\n';

    std::cout
        << "[PASS] model metrics row count valid\n";

    std::cout
        << "[INFO] effective FPS: "
        << summary.effectiveFps
        << '\n';

    std::cout
        << "[INFO] P95 frame ms: "
        << summary.p95TotalMs
        << '\n';

    std::cout
        << "[PASS] V4 multi-model video test complete\n";

    return EXIT_SUCCESS;
}