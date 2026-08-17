#include "inference/qnn_backend.hpp"
#include "models/face_detection_model.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

bool resultIsFinite(
    const models::FaceDetectionResult& result
)
{
    if (!std::isfinite(
            result.score
        )) {

        return false;
    }

    for (float value :
         result.bbox) {

        if (!std::isfinite(
                value
            )) {

            return false;
        }
    }

    for (float value :
         result.landmarks) {

        if (!std::isfinite(
                value
            )) {

            return false;
        }
    }

    return true;
}

// Same inclusive-coordinate IoU used by the
// Python postprocess NMS.
float calculateIoU(
    const std::array<float, 4>& lhs,
    const std::array<float, 4>& rhs
)
{
    const float lhsArea =
        (
            lhs[2]
            -
            lhs[0]
            +
            1.0F
        )
        *
        (
            lhs[3]
            -
            lhs[1]
            +
            1.0F
        );

    const float rhsArea =
        (
            rhs[2]
            -
            rhs[0]
            +
            1.0F
        )
        *
        (
            rhs[3]
            -
            rhs[1]
            +
            1.0F
        );

    const float xx1 =
        std::max(
            lhs[0],
            rhs[0]
        );

    const float yy1 =
        std::max(
            lhs[1],
            rhs[1]
        );

    const float xx2 =
        std::min(
            lhs[2],
            rhs[2]
        );

    const float yy2 =
        std::min(
            lhs[3],
            rhs[3]
        );

    const float width =
        std::max(
            0.0F,
            xx2
                -
                xx1
                +
                1.0F
        );

    const float height =
        std::max(
            0.0F,
            yy2
                -
                yy1
                +
                1.0F
        );

    const float intersection =
        width
        *
        height;

    const float denominator =
        lhsArea
        +
        rhsArea
        -
        intersection;

    if (denominator <= 0.0F) {
        return 0.0F;
    }

    return
        intersection
        /
        denominator;
}

void printResult(
    const models::FaceDetectionResult& result,
    std::size_t index
)
{
    std::cout
        << "[INFO] face["
        << index
        << "]\n";

    std::cout
        << std::fixed
        << std::setprecision(4);

    std::cout
        << "       score: "
        << result.score
        << '\n';

    std::cout
        << "       bbox: ["
        << result.bbox[0]
        << ", "
        << result.bbox[1]
        << ", "
        << result.bbox[2]
        << ", "
        << result.bbox[3]
        << "]\n";

    std::cout
        << "       landmarks: [";

    for (std::size_t i = 0;
         i < result.landmarks.size();
         ++i) {

        if (i > 0) {
            std::cout << ", ";
        }

        std::cout
            << result.landmarks[i];
    }

    std::cout
        << "]\n"
        << std::defaultfloat;
}

} // namespace

int main(
    int argc,
    char** argv
)
{
    if (argc != 3) {

        std::cerr
            << "Usage:\n"
            << "  "
            << argv[0]
            << " <image.jpg> <result.jpg>\n";

        return 1;
    }

    constexpr float SCORE_THRESHOLD =
        0.5F;

    constexpr float NMS_THRESHOLD =
        0.4F;

    const char* backendPath =
        std::getenv(
            "QNN_BACKEND_PATH"
        );

    const char* modelPath =
        std::getenv(
            "QNN_FACE_DETECTION_MODEL_PATH"
        );

    if (backendPath == nullptr) {

        std::cerr
            << "[ERROR] QNN_BACKEND_PATH is not set\n";

        return 1;
    }

    if (modelPath == nullptr) {

        std::cerr
            << "[ERROR] QNN_FACE_DETECTION_MODEL_PATH is not set\n";

        return 1;
    }

    const char* imagePath =
        argv[1];

    const char* resultPath =
        argv[2];

    // =====================================================
    // Input image
    // =====================================================

    cv::Mat image =
        cv::imread(
            imagePath,
            cv::IMREAD_COLOR
        );

    if (image.empty()) {

        std::cerr
            << "[ERROR] Cannot read image: "
            << imagePath
            << '\n';

        return 1;
    }

    std::cout
        << "[INFO] image: "
        << imagePath
        << '\n';

    std::cout
        << "[INFO] image size: "
        << image.cols
        << "x"
        << image.rows
        << '\n';

    // =====================================================
    // QNN backend
    // =====================================================

    inference::QnnBackend backend;

    if (!backend.loadLibrary(
            backendPath
        )) {

        std::cerr
            << "[ERROR] loadLibrary: "
            << backend.lastError()
            << '\n';

        return 1;
    }

    if (!backend.loadProviders()) {

        std::cerr
            << "[ERROR] loadProviders: "
            << backend.lastError()
            << '\n';

        return 1;
    }

    if (!backend.selectInterface()) {

        std::cerr
            << "[ERROR] selectInterface: "
            << backend.lastError()
            << '\n';

        return 1;
    }

    if (!backend.createBackend()) {

        std::cerr
            << "[ERROR] createBackend: "
            << backend.lastError()
            << '\n';

        return 1;
    }

    if (!backend.createDevice()) {

        std::cerr
            << "[ERROR] createDevice: "
            << backend.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] QNN backend ready\n";

    // =====================================================
    // Face model
    // =====================================================

    models::FaceDetectionModel model(
        backend
    );

    if (!model.initialize(
            modelPath
        )) {

        std::cerr
            << "[ERROR] initialize: "
            << model.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] FaceDetectionModel initialized\n";

    // =====================================================
    // Complete detection:
    //
    // preprocess
    // infer
    // decode 8/16/32
    // threshold
    // NMS
    // map back to image
    // =====================================================

    std::vector<
        models::FaceDetectionResult
    > results;

    if (!model.detect(
            image,
            results,
            SCORE_THRESHOLD,
            NMS_THRESHOLD
        )) {

        std::cerr
            << "[ERROR] detect: "
            << model.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] FaceDetectionModel detect succeeded\n";

    std::cout
        << "[INFO] thresholds:\n"
        << "       score: "
        << SCORE_THRESHOLD
        << '\n'
        << "       nms: "
        << NMS_THRESHOLD
        << '\n';

    std::cout
        << "[INFO] faces after NMS: "
        << results.size()
        << '\n';

    // =====================================================
    // Validate final results
    // =====================================================

    float previousScore =
        2.0F;

    for (std::size_t i = 0;
         i < results.size();
         ++i) {

        const auto& result =
            results[i];

        if (!resultIsFinite(
                result
            )) {

            std::cerr
                << "[ERROR] face["
                << i
                << "] contains non-finite values\n";

            return 1;
        }

        if (result.score <
            SCORE_THRESHOLD) {

            std::cerr
                << "[ERROR] face["
                << i
                << "] is below score threshold\n";

            return 1;
        }

        if (result.score >
            previousScore) {

            std::cerr
                << "[ERROR] final results are not "
                << "sorted by confidence\n";

            return 1;
        }

        previousScore =
            result.score;

        // BBox must already be clipped to original image.
        if (result.bbox[0] < 0.0F ||
            result.bbox[1] < 0.0F ||
            result.bbox[2] >
                static_cast<float>(
                    image.cols - 1
                ) ||
            result.bbox[3] >
                static_cast<float>(
                    image.rows - 1
                )) {

            std::cerr
                << "[ERROR] face["
                << i
                << "] bbox outside original image\n";

            return 1;
        }
    }

    std::cout
        << "[PASS] final coordinates mapped "
        << "to original image\n";

    std::cout
        << "[PASS] final detections sorted "
        << "by descending score\n";

    // =====================================================
    // Validate NMS result:
    //
    // No pair of kept boxes should have IoU > 0.4.
    // =====================================================

    for (std::size_t i = 0;
         i < results.size();
         ++i) {

        for (std::size_t j = i + 1;
             j < results.size();
             ++j) {

            const float iou =
                calculateIoU(
                    results[i].bbox,
                    results[j].bbox
                );

            if (iou >
                NMS_THRESHOLD + 1e-5F) {

                std::cerr
                    << "[ERROR] NMS validation failed: "
                    << "face["
                    << i
                    << "] vs face["
                    << j
                    << "] IoU="
                    << iou
                    << '\n';

                return 1;
            }
        }
    }

    std::cout
        << "[PASS] NMS IoU validation succeeded\n";

    // =====================================================
    // Print detections
    // =====================================================

    for (std::size_t i = 0;
         i < results.size();
         ++i) {

        printResult(
            results[i],
            i
        );
    }

    // =====================================================
    // Render final image
    // =====================================================

    cv::Mat rendered =
        model.renderDetections(
            image,
            results
        );

    if (rendered.empty()) {

        std::cerr
            << "[ERROR] rendered image is empty\n";

        return 1;
    }

    if (!cv::imwrite(
            resultPath,
            rendered
        )) {

        std::cerr
            << "[ERROR] Cannot save result image: "
            << resultPath
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] rendered image saved\n";

    std::cout
        << "[INFO] result: "
        << resultPath
        << '\n';

    std::cout
        << "[PASS] 5H.5 NMS complete\n";

    std::cout
        << "[PASS] 5H.6 final SCRFD "
        << "postprocess complete\n";

    std::cout
        << "[PASS] FaceDetectionModel "
        << "end-to-end pipeline complete\n";

    return 0;
}