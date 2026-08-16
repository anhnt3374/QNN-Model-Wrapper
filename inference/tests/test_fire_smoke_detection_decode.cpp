#include "inference/qnn_backend.hpp"
#include "models/fire_smoke_detection_model.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

bool parseAnchors(
    const std::string& text,
    std::array<
        models::FireSmokeAnchor,
        3
    >& anchors
)
{
    std::stringstream stream(
        text
    );

    std::vector<float>
        values;

    std::string token;

    while (std::getline(
        stream,
        token,
        ','
    )) {

        if (token.empty()) {
            continue;
        }

        try {

            values.push_back(
                std::stof(
                    token
                )
            );
        }
        catch (...) {

            return false;
        }
    }

    if (values.size() != 6) {

        return false;
    }

    for (std::size_t i = 0;
         i < 3;
         ++i) {

        anchors[i].width =
            values[
                i * 2
            ];

        anchors[i].height =
            values[
                i * 2 + 1
            ];

        if (!std::isfinite(
                anchors[i].width
            ) ||
            !std::isfinite(
                anchors[i].height
            ) ||
            anchors[i].width <= 0.0F ||
            anchors[i].height <= 0.0F) {

            return false;
        }
    }

    return true;
}

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

    const float intersectionWidth =
        std::max(
            0.0F,
            xx2 - xx1
        );

    const float intersectionHeight =
        std::max(
            0.0F,
            yy2 - yy1
        );

    const float intersection =
        intersectionWidth
        *
        intersectionHeight;

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

bool resultValid(
    const models::FireSmokeDetectionResult& result,
    int imageWidth,
    int imageHeight,
    float confidenceThreshold
)
{
    if (result.classId < 0 ||
        result.classId > 1) {

        return false;
    }

    if (!std::isfinite(
            result.score
        ) ||
        result.score <
            confidenceThreshold) {

        return false;
    }

    for (const float value :
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
            ) ||
        y2 >
            static_cast<float>(
                imageHeight - 1
            )) {

        return false;
    }

    return true;
}

const char* className(
    int classId
)
{
    switch (classId) {

    case 0:
        return "smoke";

    case 1:
        return "fire";

    default:
        return "unknown";
    }
}

} // namespace

int main(
    int argc,
    char** argv
)
{
    if (argc != 6) {

        std::cerr
            << "Usage:\n"
            << "  "
            << argv[0]
            << " <firesmoke.jpg>"
            << " <result.jpg>"
            << " \"s4_w1,h1,w2,h2,w3,h3\""
            << " \"s8_w1,h1,w2,h2,w3,h3\""
            << " \"s16_w1,h1,w2,h2,w3,h3\"\n";

        return 1;
    }

    constexpr float CONFIDENCE_THRESHOLD =
        0.25F;

    constexpr float NMS_THRESHOLD =
        0.45F;

    const char* backendPath =
        std::getenv(
            "QNN_BACKEND_PATH"
        );

    const char* modelPath =
        std::getenv(
            "QNN_FIRE_SMOKE_MODEL_PATH"
        );

    if (backendPath == nullptr) {

        std::cerr
            << "[ERROR] QNN_BACKEND_PATH is not set\n";

        return 1;
    }

    if (modelPath == nullptr) {

        std::cerr
            << "[ERROR] QNN_FIRE_SMOKE_MODEL_PATH is not set\n";

        return 1;
    }

    const char* imagePath =
        argv[1];

    const char* renderPath =
        argv[2];

    // =====================================================
    // Anchors
    // =====================================================

    models::FireSmokeAnchors anchors;

    if (!parseAnchors(
            argv[3],
            anchors.s4
        )) {

        std::cerr
            << "[ERROR] Invalid s4 anchors\n";

        return 1;
    }

    if (!parseAnchors(
            argv[4],
            anchors.s8
        )) {

        std::cerr
            << "[ERROR] Invalid s8 anchors\n";

        return 1;
    }

    if (!parseAnchors(
            argv[5],
            anchors.s16
        )) {

        std::cerr
            << "[ERROR] Invalid s16 anchors\n";

        return 1;
    }

    // =====================================================
    // Image
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
        ) ||
        !backend.loadProviders() ||
        !backend.selectInterface() ||
        !backend.createBackend() ||
        !backend.createDevice()) {

        std::cerr
            << "[ERROR] Backend: "
            << backend.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] QNN backend ready\n";

    // =====================================================
    // FireSmokeDetectionModel
    // =====================================================

    models::FireSmokeDetectionModel model(
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
        << "[PASS] FireSmokeDetectionModel initialized\n";

    if (!model.setAnchors(
            anchors
        )) {

        std::cerr
            << "[ERROR] setAnchors: "
            << model.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] FireSmoke anchors configured\n";

    // =====================================================
    // Complete detect API
    // =====================================================

    std::vector<
        models::FireSmokeDetectionResult
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
        << "[PASS] FireSmokeDetectionModel detect succeeded\n";

    // Outputs still contain the current inference,
    // therefore decode once without NMS for count comparison.

    std::vector<
        models::FireSmokeDetectionProposal
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
        << "[PASS] class-aware NMS did not increase count\n";

    // =====================================================
    // Validate final detections
    // =====================================================

    float previousScore =
        std::numeric_limits<float>::infinity();

    for (std::size_t i = 0;
         i < results.size();
         ++i) {

        if (!resultValid(
                results[i],
                image.cols,
                image.rows,
                CONFIDENCE_THRESHOLD
            )) {

            std::cerr
                << "[ERROR] Invalid final detection at index "
                << i
                << '\n';

            return 1;
        }

        // Final class-aware NMS output is globally sorted
        // by confidence descending.

        if (results[i].score >
            previousScore) {

            std::cerr
                << "[ERROR] Final detections are not "
                << "sorted by confidence\n";

            return 1;
        }

        previousScore =
            results[i].score;
    }

    std::cout
        << "[PASS] final detection geometry valid\n";

    std::cout
        << "[PASS] final detections sorted by confidence\n";

    // =====================================================
    // Validate CLASS-AWARE NMS.
    //
    // Only boxes of the SAME class must satisfy:
    //
    // IoU <= 0.45
    //
    // A smoke box and fire box are allowed to overlap.
    // =====================================================

    for (std::size_t i = 0;
         i < results.size();
         ++i) {

        for (std::size_t j = i + 1;
             j < results.size();
             ++j) {

            if (results[i].classId !=
                results[j].classId) {

                continue;
            }

            const float iou =
                calculateIoU(
                    results[i].bbox,
                    results[j].bbox
                );

            if (iou >
                NMS_THRESHOLD + 1e-5F) {

                std::cerr
                    << "[ERROR] Class-aware NMS validation failed: "
                    << className(
                           results[i].classId
                       )
                    << " IoU="
                    << iou
                    << '\n';

                return 1;
            }
        }
    }

    std::cout
        << "[PASS] class-aware NMS IoU validation succeeded\n";

    // =====================================================
    // Summary
    // =====================================================

    std::size_t smokeCount =
        0;

    std::size_t fireCount =
        0;

    for (const auto& result :
         results) {

        if (result.classId == 0) {

            ++smokeCount;
        }
        else if (result.classId == 1) {

            ++fireCount;
        }
    }

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

    std::cout
        << "       smoke: "
        << smokeCount
        << '\n';

    std::cout
        << "       fire: "
        << fireCount
        << '\n';

    for (std::size_t i = 0;
         i < results.size();
         ++i) {

        const auto& result =
            results[i];

        std::cout
            << std::fixed
            << std::setprecision(4)
            << "[INFO] detection["
            << i
            << "] "
            << className(
                   result.classId
               )
            << " score="
            << result.score
            << " bbox=("
            << result.bbox[0]
            << ", "
            << result.bbox[1]
            << ", "
            << result.bbox[2]
            << ", "
            << result.bbox[3]
            << ")\n"
            << std::defaultfloat;
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
            << "[ERROR] Rendered image is empty\n";

        return 1;
    }

    if (!cv::imwrite(
            renderPath,
            rendered
        )) {

        std::cerr
            << "[ERROR] Cannot save result: "
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
        << "[PASS] 7F.5 FireSmokeDetectionModel "
        << "class-aware NMS + render complete\n";

    std::cout
        << "[PASS] FireSmokeDetectionModel "
        << "end-to-end image pipeline complete\n";

    return 0;
}