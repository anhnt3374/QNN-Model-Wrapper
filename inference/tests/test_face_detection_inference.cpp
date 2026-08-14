#include "inference/qnn_backend.hpp"
#include "models/face_detection_model.hpp"

#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

bool proposalIsFinite(
    const models::FaceDetectionProposal& proposal
)
{
    if (!std::isfinite(
            proposal.score
        )) {

        return false;
    }

    for (float value :
         proposal.bbox) {

        if (!std::isfinite(
                value
            )) {

            return false;
        }
    }

    for (float value :
         proposal.landmarks) {

        if (!std::isfinite(
                value
            )) {

            return false;
        }
    }

    return true;
}

void printProposal(
    const models::FaceDetectionProposal& proposal,
    std::size_t index
)
{
    std::cout
        << "[INFO] proposal["
        << index
        << "]\n";

    std::cout
        << std::fixed
        << std::setprecision(4);

    std::cout
        << "       score: "
        << proposal.score
        << '\n';

    std::cout
        << "       bbox: ["
        << proposal.bbox[0]
        << ", "
        << proposal.bbox[1]
        << ", "
        << proposal.bbox[2]
        << ", "
        << proposal.bbox[3]
        << "]\n";

    std::cout
        << "       landmarks: [";

    for (std::size_t i = 0;
         i < proposal.landmarks.size();
         ++i) {

        if (i > 0) {
            std::cout << ", ";
        }

        std::cout
            << proposal.landmarks[i];
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
    if (argc != 2) {

        std::cerr
            << "Usage:\n"
            << "  "
            << argv[0]
            << " <image.jpg>\n";

        return 1;
    }

    const char* backendPath =
        std::getenv(
            "QNN_BACKEND_PATH"
        );

    const char* modelPath =
        std::getenv(
            "QNN_MODEL_PATH"
        );

    if (backendPath == nullptr) {

        std::cerr
            << "[ERROR] QNN_BACKEND_PATH is not set\n";

        return 1;
    }

    if (modelPath == nullptr) {

        std::cerr
            << "[ERROR] QNN_MODEL_PATH is not set\n";

        return 1;
    }

    // =====================================================
    // Real image
    // =====================================================

    const char* imagePath =
        argv[1];

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
    // SCRFD model
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
    // Preprocess
    // =====================================================

    if (!model.preprocess(
            image
        )) {

        std::cerr
            << "[ERROR] preprocess: "
            << model.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] real image preprocessed\n";

    // =====================================================
    // Inference
    // =====================================================

    if (!model.infer()) {

        std::cerr
            << "[ERROR] infer: "
            << model.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] FaceDetectionModel inference succeeded\n";

    // =====================================================
    // 5H.3-A
    //
    // threshold = 0
    //
    // Every stride-8 candidate should be returned.
    //
    // 80 * 80 * 2 = 12800
    // =====================================================

    std::vector<
        models::FaceDetectionProposal
    > allStride8;

    if (!model.decodeStride8(
            allStride8,
            0.0F
        )) {

        std::cerr
            << "[ERROR] decodeStride8(0.0): "
            << model.lastError()
            << '\n';

        return 1;
    }

    if (allStride8.size() != 12800) {

        std::cerr
            << "[ERROR] stride-8 candidate count mismatch. "
            << "expected=12800, actual="
            << allStride8.size()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] stride-8 produced 12800 candidates\n";

    // =====================================================
    // Validate values
    // =====================================================

    for (const auto& proposal :
         allStride8) {

        if (!proposalIsFinite(
                proposal
            )) {

            std::cerr
                << "[ERROR] stride-8 proposal "
                << "contains non-finite values\n";

            return 1;
        }
    }

    std::cout
        << "[PASS] stride-8 decoded values are finite\n";

    // =====================================================
    // 5H.3-B
    //
    // Same threshold used by working Python postprocess.
    // =====================================================

    constexpr float SCORE_THRESHOLD =
        0.5F;

    std::vector<
        models::FaceDetectionProposal
    > filteredStride8;

    if (!model.decodeStride8(
            filteredStride8,
            SCORE_THRESHOLD
        )) {

        std::cerr
            << "[ERROR] decodeStride8(0.5): "
            << model.lastError()
            << '\n';

        return 1;
    }

    for (const auto& proposal :
         filteredStride8) {

        if (proposal.score <
            SCORE_THRESHOLD) {

            std::cerr
                << "[ERROR] proposal below "
                << "score threshold\n";

            return 1;
        }

        if (!proposalIsFinite(
                proposal
            )) {

            std::cerr
                << "[ERROR] filtered proposal "
                << "contains non-finite values\n";

            return 1;
        }
    }

    std::cout
        << "[PASS] stride-8 score threshold applied\n";

    std::cout
        << "[INFO] stride-8 candidates:\n";

    std::cout
        << "       before threshold: "
        << allStride8.size()
        << '\n';

    std::cout
        << "       after threshold 0.5: "
        << filteredStride8.size()
        << '\n';

    // =====================================================
    // Print first few proposals for manual comparison
    // with Python if needed.
    // =====================================================

    const std::size_t printCount =
        filteredStride8.size() < 5
            ? filteredStride8.size()
            : 5;

    for (std::size_t i = 0;
         i < printCount;
         ++i) {

        printProposal(
            filteredStride8[i],
            i
        );
    }

    std::cout
        << "[PASS] 5H.3 SCRFD stride-8 "
        << "decode test complete\n";

    return 0;
}