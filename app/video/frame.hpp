#pragma once

#include <opencv2/core/mat.hpp>

#include <cstdint>

namespace video {

struct Frame {
    uint64_t frameId = 0;

    // Timestamp of this frame in the source timeline.
    //
    // For video file:
    //     derived from frame index / FPS.
    //
    // For RTSP later:
    //     can be replaced by PTS / capture timestamp.
    int64_t captureTimestampUs = 0;

    cv::Mat image;
};

} // namespace video