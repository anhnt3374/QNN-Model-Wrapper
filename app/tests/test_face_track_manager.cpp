#include "tracking/face_track_manager.hpp"

#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

bool require(
    bool condition,
    const char* message
)
{
    if (!condition) {

        std::cerr
            << "[FAIL] "
            << message
            << '\n';

        return false;
    }

    std::cout
        << "[PASS] "
        << message
        << '\n';

    return true;
}

} // namespace


int main()
{
    tracking::FaceTrackManager tracker(
        0.30F,
        2
    );

    // =====================================================
    // Frame 0
    // one new face
    // =====================================================

    std::vector<tracking::FaceTrackDetection> frame0 = {
        {
            0,
            cv::Rect2f(
                100.0F,
                100.0F,
                100.0F,
                100.0F
            ),
            0.95F
        }
    };

    const auto result0 =
        tracker.update(
            0,
            frame0
        );

    if (!require(
            result0.size() == 1,
            "frame0 creates one association"
        )) {

        return EXIT_FAILURE;
    }

    const uint64_t firstTrackId =
        result0[0].trackId;

    if (!require(
            result0[0].isNewTrack,
            "first detection creates new track"
        )) {

        return EXIT_FAILURE;
    }

    // =====================================================
    // Frame 1
    //
    // Box moves slightly.
    // IoU remains high enough.
    // Must keep same ID.
    // =====================================================

    std::vector<tracking::FaceTrackDetection> frame1 = {
        {
            0,
            cv::Rect2f(
                108.0F,
                103.0F,
                100.0F,
                100.0F
            ),
            0.96F
        }
    };

    const auto result1 =
        tracker.update(
            1,
            frame1
        );

    if (!require(
            result1.size() == 1,
            "frame1 creates one association"
        )) {

        return EXIT_FAILURE;
    }

    if (!require(
            result1[0].trackId ==
                firstTrackId,
            "nearby face keeps same track id"
        )) {

        return EXIT_FAILURE;
    }

    if (!require(
            !result1[0].isNewTrack,
            "matched track is not new"
        )) {

        return EXIT_FAILURE;
    }

    // =====================================================
    // Frame 2
    //
    // A second far-away face appears.
    // =====================================================

    std::vector<tracking::FaceTrackDetection> frame2 = {
        {
            0,
            cv::Rect2f(
                115.0F,
                105.0F,
                100.0F,
                100.0F
            ),
            0.95F
        },
        {
            1,
            cv::Rect2f(
                500.0F,
                300.0F,
                90.0F,
                90.0F
            ),
            0.91F
        }
    };

    const auto result2 =
        tracker.update(
            2,
            frame2
        );

    if (!require(
            result2.size() == 2,
            "second face produces second association"
        )) {

        return EXIT_FAILURE;
    }

    uint64_t secondTrackId =
        0;

    for (const auto& association :
         result2) {

        if (association.detectionIndex == 1) {

            secondTrackId =
                association.trackId;

            if (!require(
                    association.isNewTrack,
                    "far-away face creates new track"
                )) {

                return EXIT_FAILURE;
            }
        }
    }

    if (!require(
            secondTrackId != 0 &&
            secondTrackId != firstTrackId,
            "two faces have different track ids"
        )) {

        return EXIT_FAILURE;
    }

    // =====================================================
    // Miss first track repeatedly.
    //
    // maxMissedUpdates = 2
    //
    // After three missed detector updates it must expire.
    // =====================================================

    std::vector<tracking::FaceTrackDetection> onlySecond = {
        {
            0,
            cv::Rect2f(
                505.0F,
                302.0F,
                90.0F,
                90.0F
            ),
            0.92F
        }
    };

    tracker.update(
        3,
        onlySecond
    );

    tracker.update(
        4,
        onlySecond
    );

    tracker.update(
        5,
        onlySecond
    );

    if (!require(
            tracker.findTrack(
                firstTrackId
            ) == nullptr,
            "expired track is removed"
        )) {

        return EXIT_FAILURE;
    }

    if (!require(
            tracker.findTrack(
                secondTrackId
            ) != nullptr,
            "visible track remains active"
        )) {

        return EXIT_FAILURE;
    }

    std::cout
        << "[PASS] V6 FaceTrackManager test complete\n";

    return EXIT_SUCCESS;
}