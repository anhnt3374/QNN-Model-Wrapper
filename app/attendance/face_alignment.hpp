#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <string>

namespace attendance {

class FaceAlignment {
public:
    static constexpr int OUTPUT_SIZE =
        112;

    bool align(
        const cv::Mat& source,
        const std::array<
            cv::Point2f,
            5
        >& landmarks,
        cv::Mat& alignedFace
    );

    const std::string&
    lastError() const noexcept;

private:
    static bool landmarksValid(
        const std::array<
            cv::Point2f,
            5
        >& landmarks,
        int width,
        int height
    );

private:
    std::string lastError_;
};

} // namespace attendance