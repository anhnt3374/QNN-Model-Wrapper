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

    if (!backend.loadLibrary(backendPath)) {
        std::cerr
            << "[ERROR] loadLibrary: "
            << backend.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] libQnnHtp.so loaded\n";

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

    return 0;
}