#include "inference/qnn_context.hpp"

#include <sstream>

namespace inference {

QnnContext::QnnContext(
    QnnBackend& backend
) noexcept
    : backend_(backend)
{
}

QnnContext::~QnnContext()
{
    shutdown();
}

bool QnnContext::create()
{
    if (!backend_.interfaceReady()) {
        lastError_ =
            "QNN interface is not ready";

        return false;
    }

    if (!backend_.backendReady()) {
        lastError_ =
            "QNN backend is not ready";

        return false;
    }

    if (!backend_.deviceReady()) {
        lastError_ =
            "QNN device is not ready";

        return false;
    }

    if (contextHandle_ != nullptr) {
        return true;
    }

    const Qnn_ErrorHandle_t result =
        backend_.interface().contextCreate(
            backend_.backendHandle(),
            backend_.deviceHandle(),
            nullptr,
            &contextHandle_
        );

    if (result != QNN_CONTEXT_NO_ERROR) {
        std::ostringstream oss;

        oss
            << "contextCreate failed. error="
            << result;

        lastError_ = oss.str();

        contextHandle_ = nullptr;

        return false;
    }

    if (contextHandle_ == nullptr) {
        lastError_ =
            "contextCreate returned null handle";

        return false;
    }

    lastError_.clear();

    return true;
}

void QnnContext::shutdown()
{
    if (contextHandle_ == nullptr) {
        return;
    }

    if (!backend_.interfaceReady()) {
        contextHandle_ = nullptr;
        return;
    }

    backend_.interface().contextFree(
        contextHandle_,
        nullptr
    );

    contextHandle_ = nullptr;
}

bool QnnContext::ready() const noexcept
{
    return contextHandle_ != nullptr;
}

Qnn_ContextHandle_t
QnnContext::handle() const noexcept
{
    return contextHandle_;
}

const std::string&
QnnContext::lastError() const noexcept
{
    return lastError_;
}

} // namespace inference