#include "inference/qnn_backend.hpp"

#include <sstream>

namespace inference {

using QnnInterfaceGetProvidersFn =
    Qnn_ErrorHandle_t (*)(
        const QnnInterface_t*** providerList,
        uint32_t* numProviders
    );

bool QnnBackend::loadLibrary(
    const std::string& backendPath
)
{
    if (!backendLibrary_.open(backendPath)) {
        lastError_ =
            backendLibrary_.lastError();

        return false;
    }

    lastError_.clear();

    return true;
}

bool QnnBackend::loadProviders()
{
    if (!backendLibrary_.isOpen()) {
        lastError_ =
            "QNN backend library is not loaded";

        return false;
    }

    auto getProviders =
        backendLibrary_
            .getSymbol<QnnInterfaceGetProvidersFn>(
                "QnnInterface_getProviders"
            );

    if (getProviders == nullptr) {
        lastError_ =
            backendLibrary_.lastError();

        return false;
    }

    providers_ = nullptr;
    providerCount_ = 0;

    const Qnn_ErrorHandle_t result =
        getProviders(
            &providers_,
            &providerCount_
        );

    if (result != QNN_SUCCESS) {
        std::ostringstream oss;

        oss
            << "QnnInterface_getProviders failed. error="
            << result;

        lastError_ = oss.str();

        return false;
    }

    if (providers_ == nullptr) {
        lastError_ =
            "Provider list is null";

        return false;
    }

    if (providerCount_ == 0) {
        lastError_ =
            "QNN backend returned zero providers";

        return false;
    }

    lastError_.clear();

    return true;
}

uint32_t
QnnBackend::providerCount() const noexcept
{
    return providerCount_;
}

const std::string&
QnnBackend::lastError() const noexcept
{
    return lastError_;
}

}