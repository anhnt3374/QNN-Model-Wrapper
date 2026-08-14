#include "inference/qnn_backend.hpp"
#include "inference/qnn_model.hpp"

#include <cstdlib>
#include <iostream>

int main()
{
    const char* backendPath =
        std::getenv("QNN_BACKEND_PATH");

    const char* modelPath =
        std::getenv("QNN_MODEL_PATH");

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

    std::cout
        << "[INFO] backend: "
        << backendPath
        << '\n';

    std::cout
        << "[INFO] model: "
        << modelPath
        << '\n';

    // =====================================================
    // QNN runtime
    // =====================================================

    inference::QnnBackend backend;

    // -----------------------------------------------------
    // 1. Load backend library
    // -----------------------------------------------------

    if (!backend.loadLibrary(backendPath)) {
        std::cerr
            << "[ERROR] loadLibrary: "
            << backend.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] QNN backend library loaded\n";

    // -----------------------------------------------------
    // 2. Load providers
    // -----------------------------------------------------

    if (!backend.loadProviders()) {
        std::cerr
            << "[ERROR] loadProviders: "
            << backend.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] QNN providers loaded\n";

    // -----------------------------------------------------
    // 3. Select interface
    // -----------------------------------------------------

    if (!backend.selectInterface()) {
        std::cerr
            << "[ERROR] selectInterface: "
            << backend.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] QNN interface selected\n";

    // -----------------------------------------------------
    // 4. Create backend
    // -----------------------------------------------------

    if (!backend.createBackend()) {
        std::cerr
            << "[ERROR] createBackend: "
            << backend.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] QNN backend created\n";

    // -----------------------------------------------------
    // 5. Create device
    // -----------------------------------------------------

    if (!backend.createDevice()) {
        std::cerr
            << "[ERROR] createDevice: "
            << backend.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] QNN device created\n";

    // =====================================================
    // QNN model
    // =====================================================

    inference::QnnModel model(backend);

    // -----------------------------------------------------
    // 6. Load model library
    // -----------------------------------------------------

    if (!model.load(modelPath)) {
        std::cerr
            << "[ERROR] model.load: "
            << model.lastError()
            << '\n';

        return 1;
    }

    if (!model.libraryReady()) {
        std::cerr
            << "[ERROR] model library is not ready\n";

        return 1;
    }

    std::cout
        << "[PASS] model .so loaded\n";

    // -----------------------------------------------------
    // 7. Verify exported QNN model symbols
    // -----------------------------------------------------

    if (!model.symbolsReady()) {
        std::cerr
            << "[ERROR] required model symbols are missing\n";

        return 1;
    }

    std::cout
        << "[PASS] QnnModel_composeGraphs found\n";

    std::cout
        << "[PASS] QnnModel_freeGraphsInfo found\n";

    // -----------------------------------------------------
    // 8. Verify complete model session
    // -----------------------------------------------------

    if (!model.ready()) {
        std::cerr
            << "[ERROR] model is not ready\n";

        return 1;
    }

    std::cout
        << "[INFO] model context handle: "
        << model.contextHandle()
        << '\n';

    std::cout
        << "[INFO] model path: "
        << model.modelPath()
        << '\n';

    std::cout
        << "[PASS] QnnModel test complete\n";

    return 0;
}