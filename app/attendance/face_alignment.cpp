#include "attendance/face_alignment.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <vector>

namespace attendance {

bool FaceAlignment::align(
    const cv::Mat& source,
    const std::array<
        cv::Point2f,
        5
    >& landmarks,
    cv::Mat& alignedFace
)
{
    lastError_.clear();

    alignedFace.release();

    if (source.empty()) {

        lastError_ =
            "FaceAlignment received empty image";

        return false;
    }

    if (!landmarksValid(
            landmarks,
            source.cols,
            source.rows
        )) {

        lastError_ =
            "Face landmarks are invalid";

        return false;
    }

    // Standard 112x112 ArcFace/InsightFace-style template:
    //
    // left eye
    // right eye
    // nose
    // left mouth
    // right mouth

    static const std::array<
        cv::Point2f,
        5
    > TARGET = {
        cv::Point2f(
            38.2946F,
            51.6963F
        ),
        cv::Point2f(
            73.5318F,
            51.5014F
        ),
        cv::Point2f(
            56.0252F,
            71.7366F
        ),
        cv::Point2f(
            41.5493F,
            92.3655F
        ),
        cv::Point2f(
            70.7299F,
            92.2041F
        )
    };

    std::vector<cv::Point2f> sourcePoints(
        landmarks.begin(),
        landmarks.end()
    );

    std::vector<cv::Point2f> targetPoints(
        TARGET.begin(),
        TARGET.end()
    );

    cv::Mat inliers;

    const cv::Mat transform =
        cv::estimateAffinePartial2D(
            sourcePoints,
            targetPoints,
            inliers,
            cv::LMEDS
        );

    if (transform.empty()) {

        lastError_ =
            "Cannot estimate face alignment transform";

        return false;
    }

    cv::warpAffine(
        source,
        alignedFace,
        transform,
        cv::Size(
            OUTPUT_SIZE,
            OUTPUT_SIZE
        ),
        cv::INTER_LINEAR,
        cv::BORDER_CONSTANT,
        cv::Scalar(
            0,
            0,
            0
        )
    );

    if (alignedFace.empty() ||
        alignedFace.cols != OUTPUT_SIZE ||
        alignedFace.rows != OUTPUT_SIZE) {

        lastError_ =
            "Face alignment returned invalid image";

        return false;
    }

    return true;
}

bool FaceAlignment::landmarksValid(
    const std::array<
        cv::Point2f,
        5
    >& landmarks,
    int width,
    int height
)
{
    for (const auto& point :
         landmarks) {

        if (!std::isfinite(
                point.x
            ) ||
            !std::isfinite(
                point.y
            )) {

            return false;
        }

        if (point.x < 0.0F ||
            point.y < 0.0F ||
            point.x >=
                static_cast<float>(width) ||
            point.y >=
                static_cast<float>(height)) {

            return false;
        }
    }

    return true;
}

const std::string&
FaceAlignment::lastError() const noexcept
{
    return lastError_;
}

} // namespace attendance