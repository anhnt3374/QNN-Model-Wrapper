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
    // 6. Load model
    // -----------------------------------------------------

    if (!model.load(modelPath)) {
        std::cerr
            << "[ERROR] model.load: "
            << model.lastError()
            << '\n';

        return 1;
    }

    if (!model.ready()) {
        std::cerr
            << "[ERROR] model is not ready\n";

        return 1;
    }

    std::cout
        << "[PASS] model .so loaded\n";

    std::cout
        << "[PASS] required model symbols found\n";

    std::cout
        << "[INFO] model context handle: "
        << model.contextHandle()
        << '\n';

    // =====================================================
    // 7. Compose graphs
    // =====================================================

    if (!model.composeGraphs()) {
        std::cerr
            << "[ERROR] composeGraphs: "
            << model.lastError()
            << '\n';

        return 1;
    }

    if (!model.graphsReady()) {
        std::cerr
            << "[ERROR] graph metadata is not ready\n";

        return 1;
    }

    std::cout
        << "[PASS] QNN graphs composed\n";

    // =====================================================
    // 8. Inspect graph count
    // =====================================================

    const uint32_t graphCount =
        model.graphCount();

    std::cout
        << "[INFO] graph count: "
        << graphCount
        << '\n';

    if (graphCount == 0) {
        std::cerr
            << "[ERROR] model returned zero graphs\n";

        return 1;
    }

    std::cout
        << "[PASS] graph metadata available\n";

    std::cout
        << "[PASS] QnnModel compose test complete\n";

    return 0;
}