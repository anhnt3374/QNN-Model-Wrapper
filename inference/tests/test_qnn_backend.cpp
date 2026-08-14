#include "inference/qnn_backend.hpp"
#include "inference/qnn_context.hpp"

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
    // 3. Select interface
    // =====================================================

    if (!backend.selectInterface()) {
        std::cerr
            << "[ERROR] selectInterface: "
            << backend.lastError()
            << '\n';

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

    std::cout
        << "[PASS] QNN device created\n";

    std::cout
        << "[INFO] device handle: "
        << backend.deviceHandle()
        << '\n';

    // =====================================================
    // 6. Create independent context
    // =====================================================

    inference::QnnContext context(backend);

    if (!context.create()) {
        std::cerr
            << "[ERROR] context.create: "
            << context.lastError()
            << '\n';

        return 1;
    }

    if (!context.ready()) {
        std::cerr
            << "[ERROR] context.ready() returned false\n";

        return 1;
    }

    std::cout
        << "[PASS] QNN context created\n";

    std::cout
        << "[INFO] context handle: "
        << context.handle()
        << '\n';

    std::cout
        << "[PASS] QnnBackend/QnnContext test complete\n";

    return 0;
}