#include "pipeline/attendance_pipeline.hpp"

#include "pipeline/face_detection_adapter.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <unordered_set>

namespace pipeline {

AttendancePipeline::AttendancePipeline(
    models::FaceDetectionModel& faceDetector,
    models::FaceEmbeddingModel& faceEmbedding,
    attendance::FaceDatabase& database,
    attendance::AttendanceEventLogger& eventLogger,
    double detectorFps,
    float recognitionThreshold,
    float minimumDetectionScore,
    float minimumFaceSize,
    double attendanceCooldownSeconds,
    double unknownRetrySeconds
)
    : faceDetector_(
          faceDetector
      ),
      faceEmbedding_(
          faceEmbedding
      ),
      database_(
          database
      ),
      eventLogger_(
          eventLogger
      ),
      detectorFps_(
          detectorFps
      ),
      recognitionThreshold_(
          recognitionThreshold
      ),
      minimumDetectionScore_(
          minimumDetectionScore
      ),
      minimumFaceSize_(
          minimumFaceSize
      )
{
    attendanceCooldownUs_ =
        static_cast<int64_t>(
            std::llround(
                attendanceCooldownSeconds
                *
                1'000'000.0
            )
        );

    unknownRetryUs_ =
        static_cast<int64_t>(
            std::llround(
                unknownRetrySeconds
                *
                1'000'000.0
            )
        );
}


void AttendancePipeline::setHtpExecutionGate(
    async_runtime::HtpExecutionGate* gate
) noexcept
{
    htpGate_ =
        gate;
}


bool AttendancePipeline::configure(
    double sourceFps,
    int width,
    int height
)
{
    lastError_.clear();

    if (!faceDetector_.ready()) {

        lastError_ =
            "FaceDetectionModel is not ready";

        return false;
    }

    if (!faceEmbedding_.ready()) {

        lastError_ =
            "FaceEmbeddingModel is not ready";

        return false;
    }

    if (!eventLogger_.isOpen()) {

        lastError_ =
            "Attendance event logger is not open";

        return false;
    }

    if (!std::isfinite(sourceFps) ||
        sourceFps <= 0.0) {

        lastError_ =
            "Invalid source FPS";

        return false;
    }

    if (width <= 0 ||
        height <= 0) {

        lastError_ =
            "Invalid video dimensions";

        return false;
    }

    if (!std::isfinite(detectorFps_) ||
        detectorFps_ <= 0.0) {

        lastError_ =
            "Invalid SCRFD FPS";

        return false;
    }

    if (detectorFps_ >
        sourceFps) {

        detectorFps_ =
            sourceFps;
    }

    detectorPeriodUs_ =
        static_cast<int64_t>(
            std::llround(
                1'000'000.0
                /
                detectorFps_
            )
        );

    if (detectorPeriodUs_ <= 0) {

        lastError_ =
            "Invalid SCRFD inference period";

        return false;
    }

    scheduleInitialized_ =
        false;

    nextDetectorTimestampUs_ =
        0;

    tracker_.reset();

    identityCache_.clear();

    lastAttendanceTimestamp_.clear();

    overlays_.clear();

    detectorInferenceCount_ =
        0;

    embeddingInferenceCount_ =
        0;

    embeddingCacheHitCount_ =
        0;

    recognitionCount_ =
        0;

    attendanceEventCount_ =
        0;

    return true;
}


bool AttendancePipeline::shouldRunDetector(
    const video::Frame& frame
)
{
    if (!scheduleInitialized_) {

        scheduleInitialized_ =
            true;

        nextDetectorTimestampUs_ =
            frame.captureTimestampUs
            +
            detectorPeriodUs_;

        return true;
    }

    if (frame.captureTimestampUs <
        nextDetectorTimestampUs_) {

        return false;
    }

    do {

        nextDetectorTimestampUs_ +=
            detectorPeriodUs_;

    } while (
        nextDetectorTimestampUs_
        <=
        frame.captureTimestampUs
    );

    return true;
}


bool AttendancePipeline::qualityGate(
    const models::FaceDetectionResult& detection,
    const cv::Size& imageSize
) const
{
    const float score =
        face_adapter::confidence(
            detection
        );

    if (score <
        minimumDetectionScore_) {

        return false;
    }

    const cv::Rect2f box =
        face_adapter::boundingBox(
            detection
        );

    if (!std::isfinite(box.x) ||
        !std::isfinite(box.y) ||
        !std::isfinite(box.width) ||
        !std::isfinite(box.height)) {

        return false;
    }

    if (box.width <
            minimumFaceSize_ ||
        box.height <
            minimumFaceSize_) {

        return false;
    }

    if (box.x >=
            imageSize.width ||
        box.y >=
            imageSize.height ||
        box.x + box.width <=
            0.0F ||
        box.y + box.height <=
            0.0F) {

        return false;
    }

    const auto points =
        face_adapter::landmarks(
            detection
        );

    for (const auto& point :
         points) {

        if (!std::isfinite(point.x) ||
            !std::isfinite(point.y)) {

            return false;
        }

        if (point.x < 0.0F ||
            point.y < 0.0F ||
            point.x >=
                static_cast<float>(
                    imageSize.width
                ) ||
            point.y >=
                static_cast<float>(
                    imageSize.height
                )) {

            return false;
        }
    }

    return true;
}


float AttendancePipeline::faceQuality(
    const models::FaceDetectionResult& detection
) const
{
    const cv::Rect2f box =
        face_adapter::boundingBox(
            detection
        );

    const float size =
        std::min(
            box.width,
            box.height
        );

    return
        size
        *
        face_adapter::confidence(
            detection
        );
}


bool AttendancePipeline::shouldRunEmbedding(
    const IdentityCache& cache,
    float quality,
    int64_t timestampUs
) const
{
    // Never embedded before.
    if (!cache.embeddingAttempted) {

        return true;
    }

    // Recognized track:
    // keep identity cached until track expires.
    if (cache.recognized) {

        return false;
    }

    // Unknown:
    // retry if current face quality becomes
    // significantly better.
    if (cache.bestQuality > 0.0F &&
        quality >
            cache.bestQuality * 1.20F) {

        return true;
    }

    // Unknown:
    // periodic retry.
    if (cache.lastEmbeddingTimestampUs >= 0 &&
        timestampUs -
            cache.lastEmbeddingTimestampUs >=
            unknownRetryUs_) {

        return true;
    }

    return false;
}


bool AttendancePipeline::runEmbeddingForTrack(
    uint64_t trackId,
    const models::FaceDetectionResult& detection,
    const video::Frame& sourceFrame,
    IdentityCache& cache,
    std::vector<
        metrics::ModelMetrics
    >& modelMetrics
)
{
    cv::Mat alignedFace;

    if (!faceAlignment_.align(
            sourceFrame.image,
            face_adapter::landmarks(
                detection
            ),
            alignedFace
        )) {

        // Alignment failure is not fatal.
        return true;
    }

    metrics::ModelMetrics metric;

    metric.frameId =
        sourceFrame.frameId;

    metric.model =
        "edgeface";

    const Clock::time_point start =
        Clock::now();

    // =====================================================
    // EdgeFace PREPROCESS
    //
    // CPU work.
    // This happens outside HTP gate.
    // =====================================================

    const Clock::time_point preStart =
        Clock::now();

    if (!faceEmbedding_.preprocess(
            alignedFace
        )) {

        lastError_ =
            "EdgeFace preprocess failed: "
            +
            faceEmbedding_.lastError();

        return false;
    }

    const Clock::time_point preEnd =
        Clock::now();

    metric.preprocessMs =
        durationMs(
            preStart,
            preEnd
        );

    // =====================================================
    // EdgeFace HTP INFERENCE
    //
    // V8:
    //
    // CPU preprocess can overlap with other branches,
    // but QNN/HTP inference passes through shared gate.
    // =====================================================

    double htpWaitMs =
        0.0;

    double inferenceMs =
        0.0;

    bool inferenceSuccess =
        false;

    if (htpGate_ != nullptr) {

        inferenceSuccess =
            htpGate_->execute(
                [&]() {

                    // IMPORTANT:
                    // EdgeFace, not SCRFD.
                    return
                        faceEmbedding_.infer();
                },
                htpWaitMs,
                inferenceMs
            );
    }
    else {

        // Backward-compatible V6/V7 behavior.
        const Clock::time_point inferStart =
            Clock::now();

        inferenceSuccess =
            faceEmbedding_.infer();

        const Clock::time_point inferEnd =
            Clock::now();

        inferenceMs =
            durationMs(
                inferStart,
                inferEnd
            );
    }

    metric.queueMs =
        htpWaitMs;

    metric.inferenceMs =
        inferenceMs;

    if (!inferenceSuccess) {

        lastError_ =
            "EdgeFace inference failed: "
            +
            faceEmbedding_.lastError();

        return false;
    }

    // =====================================================
    // EdgeFace POSTPROCESS
    //
    // CPU work.
    // Outside HTP gate.
    // =====================================================

    models::FaceEmbeddingResult embedding;

    const Clock::time_point postStart =
        Clock::now();

    if (!faceEmbedding_.postprocess(
            embedding
        )) {

        lastError_ =
            "EdgeFace postprocess failed: "
            +
            faceEmbedding_.lastError();

        return false;
    }

    const Clock::time_point postEnd =
        Clock::now();

    metric.postprocessMs =
        durationMs(
            postStart,
            postEnd
        );

    metric.resultCount =
        1;

    ++embeddingInferenceCount_;

    const float quality =
        faceQuality(
            detection
        );

    cache.embeddingAttempted =
        true;

    cache.lastEmbeddingTimestampUs =
        sourceFrame.captureTimestampUs;

    cache.bestQuality =
        std::max(
            cache.bestQuality,
            quality
        );

    // =====================================================
    // DATABASE SEARCH
    // =====================================================

    const attendance::FaceMatch match =
        database_.search(
            embedding.values,
            recognitionThreshold_
        );

    if (match.matched) {

        cache.recognized =
            true;

        cache.personId =
            match.personId;

        cache.name =
            match.name;

        cache.phone =
            match.phone;

        cache.similarity =
            match.similarity;

        ++recognitionCount_;

        if (shouldEmitAttendance(
                match.personId,
                sourceFrame.captureTimestampUs
            )) {

            attendance::AttendanceEvent event;

            event.frameId =
                sourceFrame.frameId;

            event.videoTimestampUs =
                sourceFrame.captureTimestampUs;

            event.personId =
                match.personId;

            event.name =
                match.name;

            event.phone =
                match.phone;

            event.similarity =
                match.similarity;

            if (!eventLogger_.write(
                    event
                )) {

                lastError_ =
                    "Cannot write attendance event: "
                    +
                    eventLogger_.lastError();

                return false;
            }

            ++attendanceEventCount_;
        }
    }
    else {

        cache.recognized =
            false;

        cache.personId.clear();

        cache.name =
            "unknown";

        cache.phone.clear();

        cache.similarity =
            match.similarity;
    }

    metric.totalMs =
        durationMs(
            start,
            Clock::now()
        );

    modelMetrics.push_back(
        metric
    );

    (void)trackId;

    return true;
}


bool AttendancePipeline::process(
    const video::Frame& sourceFrame,
    video::Frame& renderFrame,
    std::vector<
        metrics::ModelMetrics
    >& modelMetrics
)
{
    lastError_.clear();

    if (sourceFrame.image.empty() ||
        renderFrame.image.empty()) {

        lastError_ =
            "AttendancePipeline received empty frame";

        return false;
    }

    // =====================================================
    // No SCRFD scheduled on this frame.
    //
    // Keep rendering cached track identities.
    // =====================================================

    if (!shouldRunDetector(
            sourceFrame
        )) {

        renderOverlays(
            renderFrame.image
        );

        return true;
    }

    // =====================================================
    // SCRFD
    // =====================================================

    metrics::ModelMetrics detectorMetric;

    detectorMetric.frameId =
        sourceFrame.frameId;

    detectorMetric.model =
        "scrfd";

    const Clock::time_point detectorStart =
        Clock::now();

    // =====================================================
    // SCRFD PREPROCESS
    //
    // CPU.
    // Outside HTP gate.
    // =====================================================

    const Clock::time_point preStart =
        Clock::now();

    if (!faceDetector_.preprocess(
            sourceFrame.image
        )) {

        lastError_ =
            "SCRFD preprocess failed: "
            +
            faceDetector_.lastError();

        return false;
    }

    const Clock::time_point preEnd =
        Clock::now();

    detectorMetric.preprocessMs =
        durationMs(
            preStart,
            preEnd
        );

    // =====================================================
    // SCRFD HTP INFERENCE
    // =====================================================

    double scrfdHtpWaitMs =
        0.0;

    double scrfdInferenceMs =
        0.0;

    bool scrfdInferenceSuccess =
        false;

    if (htpGate_ != nullptr) {

        scrfdInferenceSuccess =
            htpGate_->execute(
                [&]() {

                    return
                        faceDetector_.infer();
                },
                scrfdHtpWaitMs,
                scrfdInferenceMs
            );
    }
    else {

        // Backward-compatible V6/V7 path.
        const Clock::time_point inferStart =
            Clock::now();

        scrfdInferenceSuccess =
            faceDetector_.infer();

        const Clock::time_point inferEnd =
            Clock::now();

        scrfdInferenceMs =
            durationMs(
                inferStart,
                inferEnd
            );
    }

    detectorMetric.queueMs =
        scrfdHtpWaitMs;

    detectorMetric.inferenceMs =
        scrfdInferenceMs;

    if (!scrfdInferenceSuccess) {

        lastError_ =
            "SCRFD inference failed: "
            +
            faceDetector_.lastError();

        return false;
    }

    // =====================================================
    // SCRFD POSTPROCESS
    // =====================================================

    std::vector<
        models::FaceDetectionResult
    > detections;

    const Clock::time_point postStart =
        Clock::now();

    if (!faceDetector_.postprocess(
            detections
        )) {

        lastError_ =
            "SCRFD postprocess failed: "
            +
            faceDetector_.lastError();

        return false;
    }

    const Clock::time_point postEnd =
        Clock::now();

    detectorMetric.postprocessMs =
        durationMs(
            postStart,
            postEnd
        );

    detectorMetric.resultCount =
        detections.size();

    detectorMetric.totalMs =
        durationMs(
            detectorStart,
            postEnd
        );

    modelMetrics.push_back(
        detectorMetric
    );

    ++detectorInferenceCount_;

    // =====================================================
    // SCRFD detections
    // → tracker input
    // =====================================================

    std::vector<
        tracking::FaceTrackDetection
    > trackDetections;

    trackDetections.reserve(
        detections.size()
    );

    for (std::size_t i = 0;
         i < detections.size();
         ++i) {

        const auto& detection =
            detections[i];

        const float score =
            face_adapter::confidence(
                detection
            );

        const cv::Rect2f box =
            face_adapter::boundingBox(
                detection
            );

        if (score <
            minimumDetectionScore_) {

            continue;
        }

        if (box.width <= 0.0F ||
            box.height <= 0.0F) {

            continue;
        }

        tracking::FaceTrackDetection item;

        item.detectionIndex =
            i;

        item.box =
            box;

        item.score =
            score;

        trackDetections.push_back(
            item
        );
    }

    // =====================================================
    // TRACK ASSOCIATION
    // =====================================================

    const auto associations =
        tracker_.update(
            sourceFrame.frameId,
            trackDetections
        );

    // =====================================================
    // Identity cache / EdgeFace policy
    // =====================================================

    for (const auto& association :
         associations) {

        if (association.detectionIndex >=
            detections.size()) {

            continue;
        }

        const auto& detection =
            detections[
                association.detectionIndex
            ];

        IdentityCache& cache =
            identityCache_[
                association.trackId
            ];

        const bool validQuality =
            qualityGate(
                detection,
                sourceFrame.image.size()
            );

        if (!validQuality) {

            continue;
        }

        const float quality =
            faceQuality(
                detection
            );

        if (shouldRunEmbedding(
                cache,
                quality,
                sourceFrame.captureTimestampUs
            )) {

            if (!runEmbeddingForTrack(
                    association.trackId,
                    detection,
                    sourceFrame,
                    cache,
                    modelMetrics
                )) {

                return false;
            }
        }

        else if (
            cache.embeddingAttempted
        ) {

            ++embeddingCacheHitCount_;
        }
    }

    cleanupIdentityCache();

    rebuildOverlays();

    renderOverlays(
        renderFrame.image
    );

    return true;
}


void AttendancePipeline::cleanupIdentityCache()
{
    std::unordered_set<
        uint64_t
    > activeIds;

    for (const auto& track :
         tracker_.tracks()) {

        activeIds.insert(
            track.trackId
        );
    }

    for (auto iterator =
             identityCache_.begin();
         iterator !=
             identityCache_.end();) {

        if (activeIds.find(
                iterator->first
            ) ==
            activeIds.end()) {

            iterator =
                identityCache_.erase(
                    iterator
                );
        }
        else {

            ++iterator;
        }
    }
}


void AttendancePipeline::rebuildOverlays()
{
    overlays_.clear();

    overlays_.reserve(
        tracker_.tracks().size()
    );

    for (const auto& track :
         tracker_.tracks()) {

        // Do not render tracks currently missed.
        if (track.missedUpdates != 0) {

            continue;
        }

        Overlay overlay;

        overlay.trackId =
            track.trackId;

        overlay.box =
            track.box;

        overlay.detectionScore =
            track.detectionScore;

        const auto iterator =
            identityCache_.find(
                track.trackId
            );

        if (iterator ==
                identityCache_.end() ||
            !iterator->second.embeddingAttempted) {

            overlay.recognized =
                false;

            overlay.name =
                "face";

            overlays_.push_back(
                std::move(
                    overlay
                )
            );

            continue;
        }

        const IdentityCache& cache =
            iterator->second;

        overlay.recognized =
            cache.recognized;

        overlay.similarity =
            cache.similarity;

        if (cache.recognized) {

            overlay.name =
                cache.name;
        }
        else {

            overlay.name =
                "unknown";
        }

        overlays_.push_back(
            std::move(
                overlay
            )
        );
    }
}


void AttendancePipeline::renderOverlays(
    cv::Mat& image
) const
{
    for (const Overlay& overlay :
         overlays_) {

        const cv::Scalar color =
            overlay.recognized
                ?
                cv::Scalar(
                    0,
                    255,
                    0
                )
                :
                cv::Scalar(
                    0,
                    255,
                    255
                );

        const int left =
            std::max(
                0,
                static_cast<int>(
                    std::round(
                        overlay.box.x
                    )
                )
            );

        const int top =
            std::max(
                0,
                static_cast<int>(
                    std::round(
                        overlay.box.y
                    )
                )
            );

        const int right =
            std::min(
                image.cols - 1,
                static_cast<int>(
                    std::round(
                        overlay.box.x
                        +
                        overlay.box.width
                    )
                )
            );

        const int bottom =
            std::min(
                image.rows - 1,
                static_cast<int>(
                    std::round(
                        overlay.box.y
                        +
                        overlay.box.height
                    )
                )
            );

        if (right <= left ||
            bottom <= top) {

            continue;
        }

        cv::rectangle(
            image,
            cv::Point(
                left,
                top
            ),
            cv::Point(
                right,
                bottom
            ),
            color,
            2
        );

        std::ostringstream label;

        label
            << "#"
            << overlay.trackId
            << " "
            << overlay.name;

        if (overlay.similarity >=
            -0.5F) {

            label
                << " "
                << std::fixed
                << std::setprecision(2)
                << overlay.similarity;
        }

        cv::putText(
            image,
            label.str(),
            cv::Point(
                left,
                std::max(
                    20,
                    top - 6
                )
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv::LINE_AA
        );
    }
}


bool AttendancePipeline::shouldEmitAttendance(
    const std::string& personId,
    int64_t videoTimestampUs
)
{
    const auto iterator =
        lastAttendanceTimestamp_.find(
            personId
        );

    if (iterator ==
        lastAttendanceTimestamp_.end()) {

        lastAttendanceTimestamp_[
            personId
        ] =
            videoTimestampUs;

        return true;
    }

    const int64_t elapsed =
        videoTimestampUs
        -
        iterator->second;

    if (elapsed <
        attendanceCooldownUs_) {

        return false;
    }

    iterator->second =
        videoTimestampUs;

    return true;
}


const std::string&
AttendancePipeline::lastError() const noexcept
{
    return
        lastError_;
}


uint64_t
AttendancePipeline::detectorInferenceCount() const noexcept
{
    return
        detectorInferenceCount_;
}


uint64_t
AttendancePipeline::embeddingInferenceCount() const noexcept
{
    return
        embeddingInferenceCount_;
}


uint64_t
AttendancePipeline::embeddingCacheHitCount() const noexcept
{
    return
        embeddingCacheHitCount_;
}


uint64_t
AttendancePipeline::recognitionCount() const noexcept
{
    return
        recognitionCount_;
}


uint64_t
AttendancePipeline::attendanceEventCount() const noexcept
{
    return
        attendanceEventCount_;
}


uint64_t
AttendancePipeline::totalFaceTracksCreated() const noexcept
{
    return
        tracker_.totalTracksCreated();
}


std::size_t
AttendancePipeline::activeFaceTrackCount() const noexcept
{
    return
        tracker_.activeTrackCount();
}


double AttendancePipeline::durationMs(
    Clock::time_point start,
    Clock::time_point end
) noexcept
{
    return
        std::chrono::duration<
            double,
            std::milli
        >(
            end - start
        ).count();
}

} // namespace pipeline