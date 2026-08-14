#include "inference/qnn_backend.hpp"

#include <cstdlib>
#include <iostream>

int main()
{
    const char* backendPath =
        std::getenv("QNN_BACKEND_PATH");

    if (backendPath == nullptr) {
        std::cerr
            << "[ERROR] QNN_BACKEND_PATH is not set\n";

        return 1;
    }

    std::cout
        << "[INFO] backend: "
        << backendPath
        << '\n';

    inference::QnnBackend backend;

    // =====================================================
    // 1. Load libQnnHtp.so
    // =====================================================

    if (!backend.loadLibrary(backendPath)) {
        std::cerr
            << "[ERROR] loadLibrary: "
            << backend.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] libQnnHtp.so loaded\n";

    // =====================================================
    // 2. Load providers
    // =====================================================

    if (!backend.loadProviders()) {
        std::cerr
            << "[ERROR] loadProviders: "
            << backend.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] QNN providers loaded\n";

    std::cout
        << "[INFO] provider count: "
        << backend.providerCount()
        << '\n';

    // =====================================================
    // 3. Select QNN interface
    // =====================================================

    if (!backend.selectInterface()) {
        std::cerr
            << "[ERROR] selectInterface: "
            << backend.lastError()
            << '\n';

        return 1;
    }

    if (!backend.interfaceReady()) {
        std::cerr
            << "[ERROR] interfaceReady() returned false\n";

        return 1;
    }

    std::cout
        << "[PASS] QNN interface selected\n";

    // =====================================================
    // 4. Create backend
    // =====================================================

    if (!backend.createBackend()) {
        std::cerr
            << "[ERROR] createBackend: "
            << backend.lastError()
            << '\n';

        return 1;
    }

    if (!backend.backendReady()) {
        std::cerr
            << "[ERROR] backendReady() returned false\n";

        return 1;
    }

    std::cout
        << "[PASS] QNN backend created\n";

    std::cout
        << "[INFO] backend handle: "
        << backend.backendHandle()
        << '\n';

    // =====================================================
    // 5. Create device
    // =====================================================

    if (!backend.createDevice()) {
        std::cerr
            << "[ERROR] createDevice: "
            << backend.lastError()
            << '\n';

        return 1;
    }

    if (!backend.deviceReady()) {
        std::cerr
            << "[ERROR] deviceReady() returned false\n";

        return 1;
    }

    std::cout
        << "[PASS] QNN device created\n";

    std::cout
        << "[INFO] device handle: "
        << backend.deviceHandle()
        << '\n';

    std::cout
        << "[PASS] QnnBackend test complete\n";

    return 0;
}