#include "inference/qnn_backend.hpp"
#include "models/person_detection_model.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

namespace {

float calculateIoU(
    const std::array<float, 4>& lhs,
    const std::array<float, 4>& rhs
)
{
    const float lhsArea =
        std::max(
            0.0F,
            lhs[2] - lhs[0]
        )
        *
        std::max(
            0.0F,
            lhs[3] - lhs[1]
        );

    const float rhsArea =
        std::max(
            0.0F,
            rhs[2] - rhs[0]
        )
        *
        std::max(
            0.0F,
            rhs[3] - rhs[1]
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

    const float interWidth =
        std::max(
            0.0F,
            xx2 - xx1
        );

    const float interHeight =
        std::max(
            0.0F,
            yy2 - yy1
        );

    const float intersection =
        interWidth
        *
        interHeight;

    const float unionArea =
        lhsArea
        +
        rhsArea
        -
        intersection;

    return
        intersection
        /
        std::max(
            unionArea,
            1e-9F
        );
}

bool resultIsValid(
    const models::PersonDetectionResult& result,
    int imageWidth,
    int imageHeight,
    float confidenceThreshold
)
{
    if (!std::isfinite(
            result.score
        )) {

        return false;
    }

    if (result.score <
        confidenceThreshold) {

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

    const float x1 =
        result.bbox[0];

    const float y1 =
        result.bbox[1];

    const float x2 =
        result.bbox[2];

    const float y2 =
        result.bbox[3];

    if (!(x2 > x1) ||
        !(y2 > y1)) {

        return false;
    }

    if (x1 < 0.0F ||
        y1 < 0.0F) {

        return false;
    }

    if (x2 >
        static_cast<float>(
            imageWidth - 1
        )) {

        return false;
    }

    if (y2 >
        static_cast<float>(
            imageHeight - 1
        )) {

        return false;
    }

    return true;
}

void printResult(
    const models::PersonDetectionResult& result,
    std::size_t index
)
{
    std::cout
        << "[INFO] person["
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
            << " <person.jpg> <result.jpg>\n";

        return 1;
    }

    constexpr float CONFIDENCE_THRESHOLD =
        0.15F;

    constexpr float NMS_THRESHOLD =
        0.60F;

    const char* backendPath =
        std::getenv(
            "QNN_BACKEND_PATH"
        );

    const char* modelPath =
        std::getenv(
            "QNN_PERSON_MODEL_PATH"
        );

    if (backendPath == nullptr) {

        std::cerr
            << "[ERROR] QNN_BACKEND_PATH is not set\n";

        return 1;
    }

    if (modelPath == nullptr) {

        std::cerr
            << "[ERROR] QNN_PERSON_MODEL_PATH is not set\n";

        return 1;
    }

    const char* imagePath =
        argv[1];

    const char* renderPath =
        argv[2];

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
    // Person detector
    // =====================================================

    models::PersonDetectionModel model(
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
        << "[PASS] PersonDetectionModel initialized\n";

    // =====================================================
    // Complete image detection
    // =====================================================

    std::vector<
        models::PersonDetectionResult
    > results;

    if (!model.detect(
            image,
            results,
            CONFIDENCE_THRESHOLD,
            NMS_THRESHOLD
        )) {

        std::cerr
            << "[ERROR] detect: "
            << model.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] PersonDetectionModel detect succeeded\n";

    // Outputs still exist after detect(), so decode once
    // without NMS to compare counts.

    std::vector<
        models::PersonDetectionProposal
    > beforeNms;

    if (!model.decode(
            beforeNms,
            CONFIDENCE_THRESHOLD
        )) {

        std::cerr
            << "[ERROR] decode after detect: "
            << model.lastError()
            << '\n';

        return 1;
    }

    if (results.size() >
        beforeNms.size()) {

        std::cerr
            << "[ERROR] NMS increased detection count\n";

        return 1;
    }

    std::cout
        << "[PASS] NMS did not increase detection count\n";

    // =====================================================
    // Validate final detections
    // =====================================================

    float previousScore =
        std::numeric_limits<float>::infinity();

    for (std::size_t i = 0;
         i < results.size();
         ++i) {

        if (!resultIsValid(
                results[i],
                image.cols,
                image.rows,
                CONFIDENCE_THRESHOLD
            )) {

            std::cerr
                << "[ERROR] invalid final detection at index "
                << i
                << '\n';

            return 1;
        }

        // NMS returns boxes in descending score order.
        if (results[i].score >
            previousScore) {

            std::cerr
                << "[ERROR] final detections are not "
                << "sorted by confidence\n";

            return 1;
        }

        previousScore =
            results[i].score;
    }

    std::cout
        << "[PASS] final detection geometry valid\n";

    std::cout
        << "[PASS] final detections sorted by score\n";

    // =====================================================
    // Validate NMS:
    //
    // every pair of kept boxes must have IoU <= 0.60.
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
                    << "person["
                    << i
                    << "] vs person["
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
    // Summary
    // =====================================================

    std::cout
        << "[INFO] detection summary:\n";

    std::cout
        << "       confidence threshold: "
        << CONFIDENCE_THRESHOLD
        << '\n';

    std::cout
        << "       NMS threshold: "
        << NMS_THRESHOLD
        << '\n';

    std::cout
        << "       before NMS: "
        << beforeNms.size()
        << '\n';

    std::cout
        << "       after NMS: "
        << results.size()
        << '\n';

    for (std::size_t i = 0;
         i < results.size();
         ++i) {

        printResult(
            results[i],
            i
        );
    }

    // =====================================================
    // Render
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
            renderPath,
            rendered
        )) {

        std::cerr
            << "[ERROR] Cannot save render image: "
            << renderPath
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] rendered image saved\n";

    std::cout
        << "[INFO] result: "
        << renderPath
        << '\n';

    std::cout
        << "[PASS] 6P.5 PersonDetectionModel "
        << "NMS + render complete\n";

    std::cout
        << "[PASS] PersonDetectionModel "
        << "end-to-end image pipeline complete\n";

    return 0;
}