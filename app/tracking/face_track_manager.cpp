#include "tracking/face_track_manager.hpp"

#include <algorithm>
#include <limits>
#include <vector>

namespace tracking {

FaceTrackManager::FaceTrackManager(
    float minimumIou,
    std::size_t maxMissedUpdates
)
    : minimumIou_(
          minimumIou
      ),
      maxMissedUpdates_(
          maxMissedUpdates
      )
{
}


std::vector<
    FaceTrackAssociation
> FaceTrackManager::update(
    uint64_t frameId,
    const std::vector<
        FaceTrackDetection
    >& detections
)
{
    std::vector<
        FaceTrackAssociation
    > associations;

    associations.reserve(
        detections.size()
    );

    std::vector<bool> trackMatched(
        tracks_.size(),
        false
    );

    std::vector<bool> detectionMatched(
        detections.size(),
        false
    );

    // =====================================================
    // Greedy global IoU matching.
    //
    // Face count is expected to be small, therefore
    // O(N*M) is sufficient for V6.
    // =====================================================

    while (true) {

        float bestIou =
            minimumIou_;

        std::size_t bestTrack =
            std::numeric_limits<
                std::size_t
            >::max();

        std::size_t bestDetection =
            std::numeric_limits<
                std::size_t
            >::max();

        for (std::size_t trackIndex = 0;
             trackIndex < tracks_.size();
             ++trackIndex) {

            if (trackMatched[
                    trackIndex
                ]) {

                continue;
            }

            for (std::size_t detectionIndex = 0;
                 detectionIndex <
                     detections.size();
                 ++detectionIndex) {

                if (detectionMatched[
                        detectionIndex
                    ]) {

                    continue;
                }

                const float overlap =
                    iou(
                        tracks_[
                            trackIndex
                        ].box,
                        detections[
                            detectionIndex
                        ].box
                    );

                if (overlap >
                    bestIou) {

                    bestIou =
                        overlap;

                    bestTrack =
                        trackIndex;

                    bestDetection =
                        detectionIndex;
                }
            }
        }

        if (bestTrack ==
                std::numeric_limits<
                    std::size_t
                >::max() ||
            bestDetection ==
                std::numeric_limits<
                    std::size_t
                >::max()) {

            break;
        }

        FaceTrack& track =
            tracks_[
                bestTrack
            ];

        const FaceTrackDetection& detection =
            detections[
                bestDetection
            ];

        track.box =
            detection.box;

        track.detectionScore =
            detection.score;

        track.lastSeenFrameId =
            frameId;

        track.missedUpdates =
            0;

        trackMatched[
            bestTrack
        ] =
            true;

        detectionMatched[
            bestDetection
        ] =
            true;

        FaceTrackAssociation association;

        association.detectionIndex =
            detection.detectionIndex;

        association.trackId =
            track.trackId;

        association.isNewTrack =
            false;

        association.iou =
            bestIou;

        associations.push_back(
            association
        );
    }

    // =====================================================
    // Existing tracks not matched.
    // =====================================================

    for (std::size_t i = 0;
         i < tracks_.size();
         ++i) {

        if (!trackMatched[i]) {

            ++tracks_[i].missedUpdates;
        }
    }

    // =====================================================
    // New detections create tracks.
    // =====================================================

    for (std::size_t i = 0;
         i < detections.size();
         ++i) {

        if (detectionMatched[i]) {

            continue;
        }

        const FaceTrackDetection& detection =
            detections[i];

        FaceTrack track;

        track.trackId =
            nextTrackId_++;

        track.box =
            detection.box;

        track.detectionScore =
            detection.score;

        track.lastSeenFrameId =
            frameId;

        track.missedUpdates =
            0;

        tracks_.push_back(
            track
        );

        ++totalTracksCreated_;

        FaceTrackAssociation association;

        association.detectionIndex =
            detection.detectionIndex;

        association.trackId =
            track.trackId;

        association.isNewTrack =
            true;

        association.iou =
            0.0F;

        associations.push_back(
            association
        );
    }

    // =====================================================
    // Remove expired tracks.
    // =====================================================

    tracks_.erase(
        std::remove_if(
            tracks_.begin(),
            tracks_.end(),
            [this](
                const FaceTrack& track
            ) {

                return
                    track.missedUpdates >
                    maxMissedUpdates_;
            }
        ),
        tracks_.end()
    );

    return associations;
}


FaceTrack*
FaceTrackManager::findTrack(
    uint64_t trackId
) noexcept
{
    for (FaceTrack& track :
         tracks_) {

        if (track.trackId ==
            trackId) {

            return &track;
        }
    }

    return nullptr;
}


const FaceTrack*
FaceTrackManager::findTrack(
    uint64_t trackId
) const noexcept
{
    for (const FaceTrack& track :
         tracks_) {

        if (track.trackId ==
            trackId) {

            return &track;
        }
    }

    return nullptr;
}


const std::vector<
    FaceTrack
>& FaceTrackManager::tracks() const noexcept
{
    return tracks_;
}


void FaceTrackManager::reset()
{
    tracks_.clear();

    nextTrackId_ =
        1;

    totalTracksCreated_ =
        0;
}


std::size_t
FaceTrackManager::activeTrackCount() const noexcept
{
    return
        tracks_.size();
}


uint64_t
FaceTrackManager::totalTracksCreated() const noexcept
{
    return
        totalTracksCreated_;
}


float FaceTrackManager::iou(
    const cv::Rect2f& a,
    const cv::Rect2f& b
) noexcept
{
    const float left =
        std::max(
            a.x,
            b.x
        );

    const float top =
        std::max(
            a.y,
            b.y
        );

    const float right =
        std::min(
            a.x + a.width,
            b.x + b.width
        );

    const float bottom =
        std::min(
            a.y + a.height,
            b.y + b.height
        );

    const float intersectionWidth =
        std::max(
            0.0F,
            right - left
        );

    const float intersectionHeight =
        std::max(
            0.0F,
            bottom - top
        );

    const float intersection =
        intersectionWidth
        *
        intersectionHeight;

    const float areaA =
        std::max(
            0.0F,
            a.width
        )
        *
        std::max(
            0.0F,
            a.height
        );

    const float areaB =
        std::max(
            0.0F,
            b.width
        )
        *
        std::max(
            0.0F,
            b.height
        );

    const float unionArea =
        areaA
        +
        areaB
        -
        intersection;

    if (unionArea <= 0.0F) {

        return 0.0F;
    }

    return
        intersection
        /
        unionArea;
}

} // namespace tracking