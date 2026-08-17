#include "inference/qnn_backend.hpp"
#include "metrics/metrics_logger.hpp"
#include "metrics/profiler.hpp"
#include "models/person_detection_model.hpp"
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

    const char* modelPath =
        std::getenv(
            "QNN_PERSON_MODEL_PATH"
        );

    if (backendPath == nullptr ||
        modelPath == nullptr) {

        std::cerr
            << "[ERROR] QNN environment missing\n";

        return EXIT_FAILURE;
    }

    inference::QnnBackend backend;

    if (!backend.loadLibrary(backendPath) ||
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

    models::PersonDetectionModel model(
        backend
    );

    if (!model.initialize(
            modelPath
        )) {

        std::cerr
            << "[ERROR] model: "
            << model.lastError()
            << '\n';

        return EXIT_FAILURE;
    }

    std::cout
        << "[PASS] PersonDetectionModel ready\n";

    std::filesystem::remove_all(
        argv[3]
    );

    metrics::MetricsLogger logger;

    if (!logger.open(
            argv[3]
        )) {

        return EXIT_FAILURE;
    }

    metrics::PipelineProfiler profiler;

    pipeline::PersonPipeline processor(
        model,
        10.0
    );

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
        &processor
    );

    if (!pipeline.run()) {

        std::cerr
            << "[ERROR] pipeline: "
            << pipeline.lastError()
            << '\n';

        return EXIT_FAILURE;
    }

    logger.close();

    if (pipeline.stats().framesRead == 0) {

        std::cerr
            << "[ERROR] zero input frames\n";

        return EXIT_FAILURE;
    }

    if (pipeline.stats().framesRead !=
        pipeline.stats().framesWritten) {

        std::cerr
            << "[ERROR] frame count mismatch\n";

        return EXIT_FAILURE;
    }

    if (processor.inferenceCount() == 0) {

        std::cerr
            << "[ERROR] no person inference executed\n";

        return EXIT_FAILURE;
    }

    if (processor.inferenceCount() >=
        pipeline.stats().framesRead) {

        std::cerr
            << "[ERROR] person model ran every frame; "
            << "sampling is not working\n";

        return EXIT_FAILURE;
    }

    const std::string modelMetrics =
        std::string(argv[3])
        +
        "/model_metrics.csv";

    const std::size_t lines =
        countLines(
            modelMetrics
        );

    if (lines !=
        processor.inferenceCount() + 1) {

        std::cerr
            << "[ERROR] model metrics rows mismatch\n";

        return EXIT_FAILURE;
    }

    std::cout
        << "[PASS] frames read/written valid\n";

    std::cout
        << "[PASS] person sampling active\n";

    std::cout
        << "[INFO] frames: "
        << pipeline.stats().framesRead
        << '\n';

    std::cout
        << "[INFO] person inferences: "
        << processor.inferenceCount()
        << '\n';

    std::cout
        << "[PASS] model_metrics rows valid\n";

    std::cout
        << "[PASS] V3 person video pipeline test complete\n";

    return EXIT_SUCCESS;
}