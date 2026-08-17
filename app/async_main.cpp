#include "async_runtime/async_metrics_logger.hpp"
#include "async_runtime/async_video_pipeline.hpp"
#include "async_runtime/branch_worker.hpp"
#include "async_runtime/htp_execution_gate.hpp"
#include "async_runtime/overlay_result_cache.hpp"

#include "attendance/attendance_event_logger.hpp"
#include "attendance/face_database.hpp"

#include "config/fire_smoke_config.hpp"

#include "inference/qnn_backend.hpp"

#include "models/face_detection_model.hpp"
#include "models/face_embedding_model.hpp"
#include "models/fire_smoke_detection_model.hpp"
#include "models/person_detection_model.hpp"

#include "pipeline/attendance_pipeline.hpp"
#include "pipeline/fire_smoke_pipeline.hpp"
#include "pipeline/person_pipeline.hpp"

#include "video/video_file_sink.hpp"
#include "video/video_file_source.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

const char* requiredEnvironment(
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
    if (argc < 3 ||
        argc > 8) {

        std::cerr
            << "Usage:\n"
            << argv[0]
            << " <input.mp4>"
            << " <output.mp4>"
            << " [metrics_dir]"
            << " [person_fps]"
            << " [firesmoke_fps]"
            << " [scrfd_fps]"
            << " [face_threshold]\n";

        return EXIT_FAILURE;
    }

    const std::string inputPath =
        argv[1];

    const std::string outputPath =
        argv[2];

    const std::string metricsDirectory =
        argc >= 4
            ? argv[3]
            : "v8_logs";

    double personFps =
        10.0;

    double fireSmokeFps =
        5.0;

    double scrfdFps =
        10.0;

    float faceThreshold =
        0.45F;

    try {

        if (argc >= 5) {

            personFps =
                std::stod(
                    argv[4]
                );
        }

        if (argc >= 6) {

            fireSmokeFps =
                std::stod(
                    argv[5]
                );
        }

        if (argc >= 7) {

            scrfdFps =
                std::stod(
                    argv[6]
                );
        }

        if (argc >= 8) {

            faceThreshold =
                std::stof(
                    argv[7]
                );
        }
    }
    catch (...) {

        std::cerr
            << "[ERROR] Invalid numeric argument\n";

        return EXIT_FAILURE;
    }

    // =====================================================
    // Environment
    // =====================================================

    const char* backendPath =
        requiredEnvironment(
            "QNN_BACKEND_PATH"
        );

    const char* personPath =
        requiredEnvironment(
            "QNN_PERSON_MODEL_PATH"
        );

    const char* fireSmokePath =
        requiredEnvironment(
            "QNN_FIRE_SMOKE_MODEL_PATH"
        );

    const char* scrfdPath =
        requiredEnvironment(
            "QNN_FACE_DETECTION_MODEL_PATH"
        );

    const char* edgeFacePath =
        requiredEnvironment(
            "QNN_FACE_EMBEDDING_MODEL_PATH"
        );

    const char* faceDbPath =
        requiredEnvironment(
            "FACE_DATABASE_PATH"
        );

    if (backendPath == nullptr ||
        personPath == nullptr ||
        fireSmokePath == nullptr ||
        scrfdPath == nullptr ||
        edgeFacePath == nullptr ||
        faceDbPath == nullptr) {

        return EXIT_FAILURE;
    }

    // =====================================================
    // FireSmoke anchors
    // =====================================================

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

    // =====================================================
    // Shared QNN backend / device
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
            << "[ERROR] QNN backend: "
            << backend.lastError()
            << '\n';

        return EXIT_FAILURE;
    }

    // =====================================================
    // Models
    // =====================================================

    models::PersonDetectionModel personModel(
        backend
    );

    if (!personModel.initialize(
            personPath
        )) {

        std::cerr
            << "[ERROR] Person: "
            << personModel.lastError()
            << '\n';

        return EXIT_FAILURE;
    }


    models::FireSmokeDetectionModel fireSmokeModel(
        backend
    );

    if (!fireSmokeModel.initialize(
            fireSmokePath
        ) ||
        !fireSmokeModel.setAnchors(
            anchors
        )) {

        std::cerr
            << "[ERROR] FireSmoke: "
            << fireSmokeModel.lastError()
            << '\n';

        return EXIT_FAILURE;
    }


    models::FaceDetectionModel faceDetector(
        backend
    );

    if (!faceDetector.initialize(
            scrfdPath
        )) {

        std::cerr
            << "[ERROR] SCRFD: "
            << faceDetector.lastError()
            << '\n';

        return EXIT_FAILURE;
    }


    models::FaceEmbeddingModel faceEmbedding(
        backend
    );

    if (!faceEmbedding.initialize(
            edgeFacePath
        )) {

        std::cerr
            << "[ERROR] EdgeFace: "
            << faceEmbedding.lastError()
            << '\n';

        return EXIT_FAILURE;
    }

    // =====================================================
    // Face database
    // =====================================================

    attendance::FaceDatabase faceDatabase;

    if (!faceDatabase.load(
            faceDbPath
        )) {

        std::cerr
            << "[ERROR] Face DB: "
            << faceDatabase.lastError()
            << '\n';

        return EXIT_FAILURE;
    }

    // =====================================================
    // Logging
    // =====================================================

    async_runtime::AsyncMetricsLogger asyncMetrics;

    if (!asyncMetrics.open(
            metricsDirectory
        )) {

        std::cerr
            << "[ERROR] Async metrics: "
            << asyncMetrics.lastError()
            << '\n';

        return EXIT_FAILURE;
    }


    attendance::AttendanceEventLogger attendanceLogger;

    if (!attendanceLogger.open(
            metricsDirectory
            +
            "/attendance_events.jsonl"
        )) {

        std::cerr
            << "[ERROR] Attendance logger: "
            << attendanceLogger.lastError()
            << '\n';

        return EXIT_FAILURE;
    }

    // =====================================================
    // Shared HTP gate
    // =====================================================

    async_runtime::HtpExecutionGate htpGate;

    // =====================================================
    // Business pipelines
    // =====================================================

    pipeline::PersonPipeline personPipeline(
        personModel,
        personFps
    );

    personPipeline.setHtpExecutionGate(
        &htpGate
    );


    pipeline::FireSmokePipeline fireSmokePipeline(
        fireSmokeModel,
        fireSmokeFps,
        5,
        3
    );

    fireSmokePipeline.setHtpExecutionGate(
        &htpGate
    );


    pipeline::AttendancePipeline attendancePipeline(
        faceDetector,
        faceEmbedding,
        faceDatabase,
        attendanceLogger,
        scrfdFps,
        faceThreshold,
        0.60F,
        48.0F,
        5.0,
        1.0
    );

    attendancePipeline.setHtpExecutionGate(
        &htpGate
    );

    // =====================================================
    // Result caches
    // =====================================================

    async_runtime::OverlayResultCache personCache;

    async_runtime::OverlayResultCache fireSmokeCache;

    async_runtime::OverlayResultCache attendanceCache;

    // =====================================================
    // Workers
    // =====================================================

    async_runtime::BranchWorker personWorker(
        "person",
        personPipeline,
        personCache,
        asyncMetrics,
        2
    );

    async_runtime::BranchWorker fireSmokeWorker(
        "firesmoke",
        fireSmokePipeline,
        fireSmokeCache,
        asyncMetrics,
        2
    );

    async_runtime::BranchWorker attendanceWorker(
        "attendance",
        attendancePipeline,
        attendanceCache,
        asyncMetrics,
        2
    );

    // =====================================================
    // Video I/O
    // =====================================================

    video::VideoFileSource source(
        inputPath
    );

    video::VideoFileSink sink(
        outputPath
    );

    async_runtime::AsyncVideoPipeline pipeline(
        source,
        sink,
        asyncMetrics,
        2
    );

    pipeline.addWorker(
        personWorker
    );

    pipeline.addWorker(
        fireSmokeWorker
    );

    pipeline.addWorker(
        attendanceWorker
    );

    pipeline.setPersonResultCache(
        &personCache
    );

    pipeline.setFireSmokeResultCache(
        &fireSmokeCache
    );

    pipeline.setAttendanceResultCache(
        &attendanceCache
    );

    std::cout
        << "========================================\n"
        << "QCS6490 VIDEO PIPELINE - V8 ASYNC\n"
        << "========================================\n"
        << "Execution          : async\n"
        << "Scheduling         : latest-frame/drop-old\n"
        << "Queue capacity     : 2\n"
        << "HTP mode           : serialized shared gate\n"
        << "Person FPS         : "
        << personFps
        << '\n'
        << "FireSmoke FPS      : "
        << fireSmokeFps
        << '\n'
        << "SCRFD FPS          : "
        << scrfdFps
        << '\n'
        << "Identity cache     : enabled\n"
        << "Input pacing       : realtime\n"
        << "========================================\n";

    // =====================================================
    // RUN
    // =====================================================

    if (!pipeline.run()) {

        std::cerr
            << "[ERROR] V8 pipeline: "
            << pipeline.lastError()
            << '\n';

        return EXIT_FAILURE;
    }

    attendanceLogger.close();

    asyncMetrics.close();

    const auto htpStats =
        htpGate.stats();

    const double averageHtpWait =
        htpStats.executionCount > 0
            ?
            htpStats.totalWaitMs
            /
            static_cast<double>(
                htpStats.executionCount
            )
            :
            0.0;

    const double averageHtpExecution =
        htpStats.executionCount > 0
            ?
            htpStats.totalExecutionMs
            /
            static_cast<double>(
                htpStats.executionCount
            )
            :
            0.0;

    std::cout
        << std::fixed
        << std::setprecision(3);

    std::cout
        << "\n========================================\n"
        << "VIDEO PIPELINE SUMMARY - V8\n"
        << "========================================\n"
        << "Captured frames        : "
        << pipeline.capturedFrames()
        << '\n'
        << "Output frames          : "
        << pipeline.outputFrames()
        << '\n'
        << "Render dropped         : "
        << pipeline.droppedRenderFrames()
        << '\n'
        << '\n'
        << "Person worker\n"
        << "----------------------------------------\n"
        << "Processed              : "
        << personWorker.processedCount()
        << '\n'
        << "Dropped                : "
        << personWorker.droppedCount()
        << '\n'
        << "Max queue depth        : "
        << personWorker.maxQueueDepth()
        << '\n'
        << '\n'
        << "FireSmoke worker\n"
        << "----------------------------------------\n"
        << "Processed              : "
        << fireSmokeWorker.processedCount()
        << '\n'
        << "Dropped                : "
        << fireSmokeWorker.droppedCount()
        << '\n'
        << "Max queue depth        : "
        << fireSmokeWorker.maxQueueDepth()
        << '\n'
        << '\n'
        << "Attendance worker\n"
        << "----------------------------------------\n"
        << "Processed              : "
        << attendanceWorker.processedCount()
        << '\n'
        << "Dropped                : "
        << attendanceWorker.droppedCount()
        << '\n'
        << "Max queue depth        : "
        << attendanceWorker.maxQueueDepth()
        << '\n'
        << '\n'
        << "Model inference counts\n"
        << "----------------------------------------\n"
        << "Person                 : "
        << personPipeline.inferenceCount()
        << '\n'
        << "FireSmoke              : "
        << fireSmokePipeline.inferenceCount()
        << '\n'
        << "SCRFD                   : "
        << attendancePipeline.detectorInferenceCount()
        << '\n'
        << "EdgeFace                : "
        << attendancePipeline.embeddingInferenceCount()
        << '\n'
        << "EdgeFace cache hits     : "
        << attendancePipeline.embeddingCacheHitCount()
        << '\n'
        << '\n'
        << "HTP execution gate\n"
        << "----------------------------------------\n"
        << "Executions              : "
        << htpStats.executionCount
        << '\n'
        << "Average HTP wait        : "
        << averageHtpWait
        << " ms\n"
        << "Maximum HTP wait        : "
        << htpStats.maxWaitMs
        << " ms\n"
        << "Average HTP inference   : "
        << averageHtpExecution
        << " ms\n"
        << "Maximum HTP inference   : "
        << htpStats.maxExecutionMs
        << " ms\n"
        << '\n'
        << "Metrics\n"
        << "----------------------------------------\n"
        << metricsDirectory
        << "/async_frame_metrics.csv\n"
        << metricsDirectory
        << "/async_model_metrics.csv\n"
        << "========================================\n";

    return EXIT_SUCCESS;
}