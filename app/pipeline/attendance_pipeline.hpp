#pragma once

#include "attendance/attendance_event_logger.hpp"
#include "attendance/face_alignment.hpp"
#include "attendance/face_database.hpp"
#include "async_runtime/htp_execution_gate.hpp"

#include "metrics/metrics.hpp"

#include "models/face_detection_model.hpp"
#include "models/face_embedding_model.hpp"

#include "pipeline/frame_processor.hpp"

#include "tracking/face_track_manager.hpp"

#include <chrono>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace pipeline {

class AttendancePipeline final
    : public FrameProcessor {
public:
    AttendancePipeline(
        models::FaceDetectionModel& faceDetector,
        models::FaceEmbeddingModel& faceEmbedding,
        attendance::FaceDatabase& database,
        attendance::AttendanceEventLogger& eventLogger,
        double detectorFps = 10.0,
        float recognitionThreshold = 0.45F,
        float minimumDetectionScore = 0.60F,
        float minimumFaceSize = 48.0F,
        double attendanceCooldownSeconds = 5.0,
        double unknownRetrySeconds = 1.0
    );

    bool configure(
        double sourceFps,
        int width,
        int height
    ) override;

    bool process(
        const video::Frame& sourceFrame,
        video::Frame& renderFrame,
        std::vector<
            metrics::ModelMetrics
        >& modelMetrics
    ) override;

    const std::string&
    lastError() const noexcept override;

    uint64_t detectorInferenceCount() const noexcept;

    uint64_t embeddingInferenceCount() const noexcept;

    uint64_t embeddingCacheHitCount() const noexcept;

    uint64_t recognitionCount() const noexcept;

    uint64_t attendanceEventCount() const noexcept;

    uint64_t totalFaceTracksCreated() const noexcept;

    std::size_t activeFaceTrackCount() const noexcept;

    void setHtpExecutionGate(
        async_runtime::HtpExecutionGate* gate
    ) noexcept;

private:
    using Clock =
        std::chrono::steady_clock;

    struct IdentityCache {
        bool embeddingAttempted =
            false;

        bool recognized =
            false;

        std::string personId;

        std::string name;

        std::string phone;

        float similarity =
            -1.0F;

        float bestQuality =
            0.0F;

        int64_t lastEmbeddingTimestampUs =
            -1;
    };


    struct Overlay {
        uint64_t trackId =
            0;

        cv::Rect2f box;

        float detectionScore =
            0.0F;

        bool recognized =
            false;

        std::string name;

        float similarity =
            -1.0F;
    };


    bool shouldRunDetector(
        const video::Frame& frame
    );

    bool qualityGate(
        const models::FaceDetectionResult& detection,
        const cv::Size& imageSize
    ) const;

    float faceQuality(
        const models::FaceDetectionResult& detection
    ) const;

    bool shouldRunEmbedding(
        const IdentityCache& cache,
        float quality,
        int64_t timestampUs
    ) const;

    bool runEmbeddingForTrack(
        uint64_t trackId,
        const models::FaceDetectionResult& detection,
        const video::Frame& sourceFrame,
        IdentityCache& cache,
        std::vector<
            metrics::ModelMetrics
        >& modelMetrics
    );

    void cleanupIdentityCache();

    void rebuildOverlays();

    void renderOverlays(
        cv::Mat& image
    ) const;

    bool shouldEmitAttendance(
        const std::string& personId,
        int64_t videoTimestampUs
    );

    static double durationMs(
        Clock::time_point start,
        Clock::time_point end
    ) noexcept;

    async_runtime::HtpExecutionGate* htpGate_ = nullptr;

private:
    models::FaceDetectionModel& faceDetector_;

    models::FaceEmbeddingModel& faceEmbedding_;

    attendance::FaceDatabase& database_;

    attendance::AttendanceEventLogger& eventLogger_;

    attendance::FaceAlignment faceAlignment_;

    tracking::FaceTrackManager tracker_{
        0.30F,
        3
    };

    double detectorFps_ =
        10.0;

    int64_t detectorPeriodUs_ =
        100000;

    bool scheduleInitialized_ =
        false;

    int64_t nextDetectorTimestampUs_ =
        0;

    float recognitionThreshold_ =
        0.45F;

    float minimumDetectionScore_ =
        0.60F;

    float minimumFaceSize_ =
        48.0F;

    int64_t attendanceCooldownUs_ =
        5'000'000;

    int64_t unknownRetryUs_ =
        1'000'000;

    std::unordered_map<
        uint64_t,
        IdentityCache
    > identityCache_;

    std::unordered_map<
        std::string,
        int64_t
    > lastAttendanceTimestamp_;

    std::vector<
        Overlay
    > overlays_;

    uint64_t detectorInferenceCount_ =
        0;

    uint64_t embeddingInferenceCount_ =
        0;

    uint64_t embeddingCacheHitCount_ =
        0;

    uint64_t recognitionCount_ =
        0;

    uint64_t attendanceEventCount_ =
        0;

    std::string lastError_;
};

} // namespace pipeline