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

bool proposalsAreSortedDescending(
    const std::vector<
        models::FaceDetectionProposal
    >& proposals
)
{
    if (proposals.size() < 2) {
        return true;
    }

    for (std::size_t i = 1;
         i < proposals.size();
         ++i) {

        if (proposals[i - 1].score <
            proposals[i].score) {

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
    // Backend
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
    // FaceDetectionModel
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
    // 5H.4-A
    //
    // threshold = 0
    //
    // Every candidate from all 3 levels must survive.
    //
    // 12800 + 3200 + 800 = 16800.
    // =====================================================

    std::vector<
        models::FaceDetectionProposal
    > allProposals;

    if (!model.decodeAll(
            allProposals,
            0.0F
        )) {

        std::cerr
            << "[ERROR] decodeAll(0.0): "
            << model.lastError()
            << '\n';

        return 1;
    }

    constexpr std::size_t EXPECTED_ALL =
        16800;

    if (allProposals.size() !=
        EXPECTED_ALL) {

        std::cerr
            << "[ERROR] all-level candidate count mismatch. "
            << "expected="
            << EXPECTED_ALL
            << ", actual="
            << allProposals.size()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] all SCRFD levels produced "
        << EXPECTED_ALL
        << " candidates\n";

    // =====================================================
    // Every decoded value should be finite.
    // =====================================================

    for (const auto& proposal :
         allProposals) {

        if (!proposalIsFinite(
                proposal
            )) {

            std::cerr
                << "[ERROR] decoded proposal contains "
                << "non-finite value\n";

            return 1;
        }
    }

    std::cout
        << "[PASS] all decoded values are finite\n";

    // =====================================================
    // decodeAll() must sort descending by score.
    // =====================================================

    if (!proposalsAreSortedDescending(
            allProposals
        )) {

        std::cerr
            << "[ERROR] proposals are not sorted "
            << "by descending score\n";

        return 1;
    }

    std::cout
        << "[PASS] proposals sorted by descending score\n";

    // =====================================================
    // 5H.4-B
    //
    // Same threshold as working Python.
    // =====================================================

    constexpr float SCORE_THRESHOLD =
        0.5F;

    std::vector<
        models::FaceDetectionProposal
    > filteredProposals;

    if (!model.decodeAll(
            filteredProposals,
            SCORE_THRESHOLD
        )) {

        std::cerr
            << "[ERROR] decodeAll(0.5): "
            << model.lastError()
            << '\n';

        return 1;
    }

    for (const auto& proposal :
         filteredProposals) {

        if (proposal.score <
            SCORE_THRESHOLD) {

            std::cerr
                << "[ERROR] proposal below score threshold\n";

            return 1;
        }

        if (!proposalIsFinite(
                proposal
            )) {

            std::cerr
                << "[ERROR] filtered proposal contains "
                << "non-finite value\n";

            return 1;
        }
    }

    if (!proposalsAreSortedDescending(
            filteredProposals
        )) {

        std::cerr
            << "[ERROR] filtered proposals are not sorted\n";

        return 1;
    }

    std::cout
        << "[PASS] score threshold 0.5 applied\n";

    std::cout
        << "[PASS] filtered proposals remain sorted\n";

    // =====================================================
    // Summary
    // =====================================================

    std::cout
        << "[INFO] SCRFD proposals:\n";

    std::cout
        << "       stride 8 candidates : 12800\n";

    std::cout
        << "       stride 16 candidates: 3200\n";

    std::cout
        << "       stride 32 candidates: 800\n";

    std::cout
        << "       total before threshold: "
        << allProposals.size()
        << '\n';

    std::cout
        << "       after threshold 0.5: "
        << filteredProposals.size()
        << '\n';

    // Print highest-confidence proposals.
    const std::size_t printCount =
        filteredProposals.size() < 5
            ? filteredProposals.size()
            : 5;

    for (std::size_t i = 0;
         i < printCount;
         ++i) {

        printProposal(
            filteredProposals[i],
            i
        );
    }

    std::cout
        << "[PASS] 5H.4 SCRFD all-level "
        << "decode test complete\n";

    return 0;
}