#pragma once

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tracking {

struct FaceTrackDetection {
    std::size_t detectionIndex =
        0;

    cv::Rect2f box;

    float score =
        0.0F;
};


struct FaceTrack {
    uint64_t trackId =
        0;

    cv::Rect2f box;

    float detectionScore =
        0.0F;

    uint64_t lastSeenFrameId =
        0;

    std::size_t missedUpdates =
        0;
};


struct FaceTrackAssociation {
    std::size_t detectionIndex =
        0;

    uint64_t trackId =
        0;

    bool isNewTrack =
        false;

    float iou =
        0.0F;
};


class FaceTrackManager {
public:
    explicit FaceTrackManager(
        float minimumIou = 0.30F,
        std::size_t maxMissedUpdates = 3
    );

    std::vector<
        FaceTrackAssociation
    > update(
        uint64_t frameId,
        const std::vector<
            FaceTrackDetection
        >& detections
    );

    FaceTrack*
    findTrack(
        uint64_t trackId
    ) noexcept;

    const FaceTrack*
    findTrack(
        uint64_t trackId
    ) const noexcept;

    const std::vector<
        FaceTrack
    >& tracks() const noexcept;

    void reset();

    std::size_t activeTrackCount() const noexcept;

    uint64_t totalTracksCreated() const noexcept;

    static float iou(
        const cv::Rect2f& a,
        const cv::Rect2f& b
    ) noexcept;

private:
    float minimumIou_ =
        0.30F;

    std::size_t maxMissedUpdates_ =
        3;

    uint64_t nextTrackId_ =
        1;

    uint64_t totalTracksCreated_ =
        0;

    std::vector<
        FaceTrack
    > tracks_;
};

} // namespace tracking