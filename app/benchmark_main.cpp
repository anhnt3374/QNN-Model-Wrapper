#include "benchmark/benchmark_report.hpp"

#include "config/fire_smoke_config.hpp"

#include "attendance/attendance_event_logger.hpp"
#include "attendance/face_database.hpp"

#include "inference/qnn_backend.hpp"

#include "metrics/metrics_logger.hpp"
#include "metrics/profiler.hpp"

#include "models/face_detection_model.hpp"
#include "models/face_embedding_model.hpp"
#include "models/fire_smoke_detection_model.hpp"
#include "models/person_detection_model.hpp"

#include "pipeline/attendance_pipeline.hpp"
#include "pipeline/composite_frame_processor.hpp"
#include "pipeline/fire_smoke_pipeline.hpp"
#include "pipeline/person_pipeline.hpp"
#include "pipeline/video_pipeline.hpp"

#include "video/video_file_sink.hpp"
#include "video/video_file_source.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

namespace {

enum class BenchmarkProfile {
    B0,
    B1,
    B2,
    B3
};


bool parseProfile(
    const std::string& text,
    BenchmarkProfile& profile
)
{
    if (text == "b0") {

        profile =
            BenchmarkProfile::B0;

        return true;
    }

    if (text == "b1") {

        profile =
            BenchmarkProfile::B1;

        return true;
    }

    if (text == "b2") {

        profile =
            BenchmarkProfile::B2;

        return true;
    }

    if (text == "b3") {

        profile =
            BenchmarkProfile::B3;

        return true;
    }

    return false;
}


bool needsQnn(
    BenchmarkProfile profile
)
{
    return
        profile !=
        BenchmarkProfile::B0;
}


bool needsPerson(
    BenchmarkProfile profile
)
{
    return
        profile ==
            BenchmarkProfile::B1
        ||
        profile ==
            BenchmarkProfile::B2
        ||
        profile ==
            BenchmarkProfile::B3;
}


bool needsFireSmoke(
    BenchmarkProfile profile
)
{
    return
        profile ==
            BenchmarkProfile::B2
        ||
        profile ==
            BenchmarkProfile::B3;
}


bool needsAttendance(
    BenchmarkProfile profile
)
{
    return
        profile ==
        BenchmarkProfile::B3;
}


const char* profileName(
    BenchmarkProfile profile
)
{
    switch (profile) {

    case BenchmarkProfile::B0:
        return "b0";

    case BenchmarkProfile::B1:
        return "b1";

    case BenchmarkProfile::B2:
        return "b2";

    case BenchmarkProfile::B3:
        return "b3";
    }

    return "unknown";
}


const char* profileDescription(
    BenchmarkProfile profile
)
{
    switch (profile) {

    case BenchmarkProfile::B0:
        return "Video only";

    case BenchmarkProfile::B1:
        return "Video + Person";

    case BenchmarkProfile::B2:
        return "Video + Person + FireSmoke";

    case BenchmarkProfile::B3:
        return "Full workload";
    }

    return "Unknown";
}


const char*
requiredEnvironment(
    const char* name
)
{
    const char* value =
        std::getenv(
            name
        );

    if (value == nullptr ||
        value[0] == '\0') {

        std::cerr
            << "[ERROR] "
            << name
            << " is not set\n";

        return nullptr;
    }

    return value;
}

} // namespace


int main(
    int argc,
    char** argv
)
{
    if (argc < 5 ||
        argc > 9) {

        std::cerr
            << "Usage:\n"
            << "  "
            << argv[0]
            << " <b0|b1|b2|b3>"
            << " <input.mp4>"
            << " <output.mp4>"
            << " <metrics_dir>"
            << " [person_fps]"
            << " [firesmoke_fps]"
            << " [scrfd_fps]"
            << " [face_threshold]\n\n"
            << "Profiles:\n"
            << "  b0  Video only\n"
            << "  b1  Video + Person\n"
            << "  b2  Video + Person + FireSmoke\n"
            << "  b3  Full workload\n";

        return EXIT_FAILURE;
    }

    BenchmarkProfile profile;

    if (!parseProfile(
            argv[1],
            profile
        )) {

        std::cerr
            << "[ERROR] Invalid benchmark profile: "
            << argv[1]
            << '\n';

        return EXIT_FAILURE;
    }

    const std::string inputPath =
        argv[2];

    const std::string outputPath =
        argv[3];

    const std::string metricsDirectory =
        argv[4];

    double personFps =
        10.0;

    double fireSmokeFps =
        5.0;

    double scrfdFps =
        10.0;

    float faceThreshold =
        0.45F;

    try {

        if (argc >= 6) {

            personFps =
                std::stod(
                    argv[5]
                );
        }

        if (argc >= 7) {

            fireSmokeFps =
                std::stod(
                    argv[6]
                );
        }

        if (argc >= 8) {

            scrfdFps =
                std::stod(
                    argv[7]
                );
        }

        if (argc >= 9) {

            faceThreshold =
                std::stof(
                    argv[8]
                );
        }
    }
    catch (...) {

        std::cerr
            << "[ERROR] Invalid numeric argument\n";

        return EXIT_FAILURE;
    }

    // =====================================================
    // Shared runtime objects
    // =====================================================

    std::unique_ptr<
        inference::QnnBackend
    > backend;

    std::unique_ptr<
        models::PersonDetectionModel
    > personModel;

    std::unique_ptr<
        models::FireSmokeDetectionModel
    > fireSmokeModel;

    std::unique_ptr<
        models::FaceDetectionModel
    > faceDetector;

    std::unique_ptr<
        models::FaceEmbeddingModel
    > faceEmbedding;


    std::unique_ptr<
        pipeline::PersonPipeline
    > personPipeline;

    std::unique_ptr<
        pipeline::FireSmokePipeline
    > fireSmokePipeline;

    std::unique_ptr<
        pipeline::AttendancePipeline
    > attendancePipeline;


    attendance::FaceDatabase faceDatabase;

    std::unique_ptr<
        attendance::AttendanceEventLogger
    > attendanceLogger;


    pipeline::CompositeFrameProcessor composite;

    pipeline::FrameProcessor* processor =
        nullptr;

    // =====================================================
    // QNN backend
    //
    // IMPORTANT:
    //
    // B0 does NOT load QNN.
    // =====================================================

    if (needsQnn(
            profile
        )) {

        const char* backendPath =
            requiredEnvironment(
                "QNN_BACKEND_PATH"
            );

        if (backendPath == nullptr) {

            return EXIT_FAILURE;
        }

        backend =
            std::make_unique<
                inference::QnnBackend
            >();

        if (!backend->loadLibrary(
                backendPath
            ) ||
            !backend->loadProviders() ||
            !backend->selectInterface() ||
            !backend->createBackend() ||
            !backend->createDevice()) {

            std::cerr
                << "[ERROR] QNN backend: "
                << backend->lastError()
                << '\n';

            return EXIT_FAILURE;
        }

        std::cout
            << "[PASS] QNN backend ready\n";
    }

    // =====================================================
    // Person
    // =====================================================

    if (needsPerson(
            profile
        )) {

        const char* modelPath =
            requiredEnvironment(
                "QNN_PERSON_MODEL_PATH"
            );

        if (modelPath == nullptr) {

            return EXIT_FAILURE;
        }

        personModel =
            std::make_unique<
                models::PersonDetectionModel
            >(
                *backend
            );

        if (!personModel->initialize(
                modelPath
            )) {

            std::cerr
                << "[ERROR] Person: "
                << personModel->lastError()
                << '\n';

            return EXIT_FAILURE;
        }

        personPipeline =
            std::make_unique<
                pipeline::PersonPipeline
            >(
                *personModel,
                personFps
            );

        composite.addProcessor(
            *personPipeline
        );
    }

    // =====================================================
    // FireSmoke
    // =====================================================

    if (needsFireSmoke(
            profile
        )) {

        const char* modelPath =
            requiredEnvironment(
                "QNN_FIRE_SMOKE_MODEL_PATH"
            );

        if (modelPath == nullptr) {

            return EXIT_FAILURE;
        }

        models::FireSmokeAnchors anchors;

        std::string anchorError;

        if (!config::loadFireSmokeAnchorsFromEnvironment(
                anchors,
                anchorError
            )) {

            std::cerr
                << "[ERROR] FireSmoke anchors: "
                << anchorError
                << '\n';

            return EXIT_FAILURE;
        }

        fireSmokeModel =
            std::make_unique<
                models::FireSmokeDetectionModel
            >(
                *backend
            );

        if (!fireSmokeModel->initialize(
                modelPath
            )) {

            std::cerr
                << "[ERROR] FireSmoke: "
                << fireSmokeModel->lastError()
                << '\n';

            return EXIT_FAILURE;
        }

        if (!fireSmokeModel->setAnchors(
                anchors
            )) {

            std::cerr
                << "[ERROR] FireSmoke anchors: "
                << fireSmokeModel->lastError()
                << '\n';

            return EXIT_FAILURE;
        }

        fireSmokePipeline =
            std::make_unique<
                pipeline::FireSmokePipeline
            >(
                *fireSmokeModel,
                fireSmokeFps,
                5,
                3
            );

        composite.addProcessor(
            *fireSmokePipeline
        );
    }

    // =====================================================
    // Attendance
    // =====================================================

    if (needsAttendance(
            profile
        )) {

        const char* detectorPath =
            requiredEnvironment(
                "QNN_FACE_DETECTION_MODEL_PATH"
            );

        const char* embeddingPath =
            requiredEnvironment(
                "QNN_FACE_EMBEDDING_MODEL_PATH"
            );

        const char* databasePath =
            requiredEnvironment(
                "FACE_DATABASE_PATH"
            );

        if (detectorPath == nullptr ||
            embeddingPath == nullptr ||
            databasePath == nullptr) {

            return EXIT_FAILURE;
        }

        faceDetector =
            std::make_unique<
                models::FaceDetectionModel
            >(
                *backend
            );

        if (!faceDetector->initialize(
                detectorPath
            )) {

            std::cerr
                << "[ERROR] SCRFD: "
                << faceDetector->lastError()
                << '\n';

            return EXIT_FAILURE;
        }

        faceEmbedding =
            std::make_unique<
                models::FaceEmbeddingModel
            >(
                *backend
            );

        if (!faceEmbedding->initialize(
                embeddingPath
            )) {

            std::cerr
                << "[ERROR] EdgeFace: "
                << faceEmbedding->lastError()
                << '\n';

            return EXIT_FAILURE;
        }

        if (!faceDatabase.load(
                databasePath
            )) {

            std::cerr
                << "[ERROR] Face DB: "
                << faceDatabase.lastError()
                << '\n';

            return EXIT_FAILURE;
        }

        attendanceLogger =
            std::make_unique<
                attendance::AttendanceEventLogger
            >();

        if (!attendanceLogger->open(
                metricsDirectory
                +
                "/attendance_events.jsonl"
            )) {

            std::cerr
                << "[ERROR] Attendance logger: "
                << attendanceLogger->lastError()
                << '\n';

            return EXIT_FAILURE;
        }

        attendancePipeline =
            std::make_unique<
                pipeline::AttendancePipeline
            >(
                *faceDetector,
                *faceEmbedding,
                faceDatabase,
                *attendanceLogger,
                scrfdFps,
                faceThreshold,
                0.60F,
                48.0F,
                5.0,
                1.0
            );

        composite.addProcessor(
            *attendancePipeline
        );
    }

    // B0: no processor at all.
    if (profile !=
        BenchmarkProfile::B0) {

        processor =
            &composite;
    }

    // =====================================================
    // Metrics
    // =====================================================

    metrics::MetricsLogger metricsLogger;

    if (!metricsLogger.open(
            metricsDirectory
        )) {

        std::cerr
            << "[ERROR] Metrics logger: "
            << metricsLogger.lastError()
            << '\n';

        return EXIT_FAILURE;
    }

    metrics::PipelineProfiler profiler(
        1000.0
    );

    // =====================================================
    // Video
    // =====================================================

    video::VideoFileSource source(
        inputPath
    );

    video::VideoFileSink sink(
        outputPath
    );

    pipeline::VideoPipeline videoPipeline(
        source,
        sink,
        &profiler,
        &metricsLogger,
        processor
    );

    // =====================================================
    // Header
    // =====================================================

    std::cout
        << "========================================\n"
        << "QCS6490 BENCHMARK - V7\n"
        << "========================================\n"
        << "Profile            : "
        << profileName(
               profile
           )
        << '\n'
        << "Workload           : "
        << profileDescription(
               profile
           )
        << '\n'
        << "Input              : "
        << inputPath
        << '\n'
        << "Output             : "
        << outputPath
        << '\n'
        << "Metrics            : "
        << metricsDirectory
        << '\n';

    if (needsPerson(
            profile
        )) {

        std::cout
            << "Person FPS         : "
            << personFps
            << '\n';
    }

    if (needsFireSmoke(
            profile
        )) {

        std::cout
            << "FireSmoke FPS      : "
            << fireSmokeFps
            << '\n';
    }

    if (needsAttendance(
            profile
        )) {

        std::cout
            << "SCRFD FPS          : "
            << scrfdFps
            << '\n'
            << "Face threshold     : "
            << faceThreshold
            << '\n'
            << "Identity cache     : enabled\n";
    }

    std::cout
        << "========================================\n";

    // =====================================================
    // RUN
    // =====================================================

    if (!videoPipeline.run()) {

        std::cerr
            << "[ERROR] Pipeline: "
            << videoPipeline.lastError()
            << '\n';

        return EXIT_FAILURE;
    }

    if (attendanceLogger) {

        attendanceLogger->close();
    }

    metricsLogger.close();

    const metrics::PipelineSummary summary =
        profiler.summary();

    // =====================================================
    // Generate benchmark report
    // =====================================================

    std::string reportError;

    if (!benchmark::BenchmarkReport::generate(
            metricsDirectory,
            profileName(
                profile
            ),
            summary,
            reportError
        )) {

        std::cerr
            << "[ERROR] Benchmark report: "
            << reportError
            << '\n';

        return EXIT_FAILURE;
    }

    // =====================================================
    // Summary
    // =====================================================

    std::cout
        << std::fixed
        << std::setprecision(3);

    std::cout
        << "\n========================================\n"
        << "BENCHMARK SUMMARY - "
        << profileName(
               profile
           )
        << "\n"
        << "========================================\n"
        << "Frames              : "
        << summary.frameCount
        << '\n'
        << "Runtime             : "
        << summary.runtimeMs
        << " ms\n"
        << "Effective FPS       : "
        << summary.effectiveFps
        << '\n'
        << '\n'
        << "Average latency\n"
        << "----------------------------------------\n"
        << "Read                : "
        << summary.averageReadMs
        << " ms\n"
        << "Process             : "
        << summary.averageProcessMs
        << " ms\n"
        << "Write               : "
        << summary.averageWriteMs
        << " ms\n"
        << "Total               : "
        << summary.averageTotalMs
        << " ms\n"
        << '\n'
        << "Frame latency\n"
        << "----------------------------------------\n"
        << "P50                 : "
        << summary.p50TotalMs
        << " ms\n"
        << "P95                 : "
        << summary.p95TotalMs
        << " ms\n"
        << "P99                 : "
        << summary.p99TotalMs
        << " ms\n"
        << "MAX                 : "
        << summary.maxTotalMs
        << " ms\n";

    if (personPipeline) {

        std::cout
            << "Person inferences   : "
            << personPipeline->inferenceCount()
            << '\n';
    }

    if (fireSmokePipeline) {

        std::cout
            << "FireSmoke inferences: "
            << fireSmokePipeline->inferenceCount()
            << '\n';
    }

    if (attendancePipeline) {

        std::cout
            << "SCRFD inferences    : "
            << attendancePipeline->
                   detectorInferenceCount()
            << '\n'
            << "EdgeFace inferences : "
            << attendancePipeline->
                   embeddingInferenceCount()
            << '\n'
            << "EdgeFace cache hits : "
            << attendancePipeline->
                   embeddingCacheHitCount()
            << '\n'
            << "Face tracks created : "
            << attendancePipeline->
                   totalFaceTracksCreated()
            << '\n';
    }

    std::cout
        << "----------------------------------------\n"
        << "Report:\n"
        << metricsDirectory
        << "/benchmark_summary.json\n"
        << "========================================\n";

    return EXIT_SUCCESS;
}