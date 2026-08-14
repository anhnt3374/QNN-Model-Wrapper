#include "inference/qnn_backend.hpp"

#include <iostream>
#include <sstream>

namespace inference {

using QnnInterfaceGetProvidersFn =
    Qnn_ErrorHandle_t (*)(
        const QnnInterface_t*** providerList,
        uint32_t* numProviders
    );

QnnBackend::~QnnBackend()
{
    shutdown();
}

bool QnnBackend::loadLibrary(
    const std::string& backendPath
)
{
    shutdown();

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

bool QnnBackend::selectInterface()
{
    if (providers_ == nullptr ||
        providerCount_ == 0) {

        lastError_ =
            "QNN providers are not loaded";

        return false;
    }

    interfaceReady_ = false;

    std::cout
        << "[INFO] application QNN API "
        << QNN_API_VERSION_MAJOR
        << "."
        << QNN_API_VERSION_MINOR
        << '\n';

    for (uint32_t i = 0;
         i < providerCount_;
         ++i) {

        const QnnInterface_t* provider =
            providers_[i];

        if (provider == nullptr) {
            std::cout
                << "[WARN] provider["
                << i
                << "] is null\n";

            continue;
        }

        const auto& coreVersion =
            provider->apiVersion.coreApiVersion;

        std::cout
            << "[INFO] provider["
            << i
            << "] core API "
            << coreVersion.major
            << "."
            << coreVersion.minor
            << "."
            << coreVersion.patch
            << '\n';

        const bool majorCompatible =
            coreVersion.major ==
            QNN_API_VERSION_MAJOR;

        const bool minorCompatible =
            coreVersion.minor >=
            QNN_API_VERSION_MINOR;

        if (!majorCompatible ||
            !minorCompatible) {

            continue;
        }

        qnnInterface_ =
            provider->QNN_INTERFACE_VER_NAME;

        interfaceReady_ = true;

        lastError_.clear();

        std::cout
            << "[INFO] selected provider["
            << i
            << "]\n";

        return true;
    }

    lastError_ =
        "No compatible QNN provider found";

    return false;
}

bool QnnBackend::createBackend()
{
    if (!interfaceReady_) {
        lastError_ =
            "QNN interface is not selected";

        return false;
    }

    if (backendHandle_ != nullptr) {
        return true;
    }

    const Qnn_ErrorHandle_t result =
        qnnInterface_.backendCreate(
            nullptr,
            nullptr,
            &backendHandle_
        );

    if (result != QNN_BACKEND_NO_ERROR) {
        std::ostringstream oss;

        oss
            << "backendCreate failed. error="
            << result;

        lastError_ = oss.str();

        backendHandle_ = nullptr;

        return false;
    }

    if (backendHandle_ == nullptr) {
        lastError_ =
            "backendCreate returned null handle";

        return false;
    }

    lastError_.clear();

    return true;
}

bool QnnBackend::createDevice()
{
    if (!interfaceReady_) {
        lastError_ =
            "QNN interface is not selected";

        return false;
    }

    if (backendHandle_ == nullptr) {
        lastError_ =
            "QNN backend is not created";

        return false;
    }

    if (deviceHandle_ != nullptr) {
        return true;
    }

    const Qnn_ErrorHandle_t result =
        qnnInterface_.deviceCreate(
            nullptr,
            nullptr,
            &deviceHandle_
        );

    if (result != QNN_DEVICE_NO_ERROR) {
        std::ostringstream oss;

        oss
            << "deviceCreate failed. error="
            << result;

        lastError_ = oss.str();

        deviceHandle_ = nullptr;

        return false;
    }

    if (deviceHandle_ == nullptr) {
        lastError_ =
            "deviceCreate returned null handle";

        return false;
    }

    lastError_.clear();

    return true;
}

void QnnBackend::shutdown()
{
    // Device phải được free trước backend.
    if (deviceHandle_ != nullptr &&
        interfaceReady_) {

        qnnInterface_.deviceFree(
            deviceHandle_
        );

        deviceHandle_ = nullptr;
    }

    if (backendHandle_ != nullptr &&
        interfaceReady_) {

        qnnInterface_.backendFree(
            backendHandle_
        );

        backendHandle_ = nullptr;
    }

    interfaceReady_ = false;

    providers_ = nullptr;
    providerCount_ = 0;

    qnnInterface_ = {};

    backendLibrary_.close();
}

uint32_t
QnnBackend::providerCount() const noexcept
{
    return providerCount_;
}

bool QnnBackend::interfaceReady() const noexcept
{
    return interfaceReady_;
}

bool QnnBackend::backendReady() const noexcept
{
    return backendHandle_ != nullptr;
}

bool QnnBackend::deviceReady() const noexcept
{
    return deviceHandle_ != nullptr;
}

Qnn_BackendHandle_t
QnnBackend::backendHandle() const noexcept
{
    return backendHandle_;
}

Qnn_DeviceHandle_t
QnnBackend::deviceHandle() const noexcept
{
    return deviceHandle_;
}

QNN_INTERFACE_VER_TYPE&
QnnBackend::interface() noexcept
{
    return qnnInterface_;
}

const std::string&
QnnBackend::lastError() const noexcept
{
    return lastError_;
}

} // namespace inference