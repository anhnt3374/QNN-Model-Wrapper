#pragma once

#include "inference/shared_library.hpp"

#include <QnnBackend.h>
#include <QnnInterface.h>

#include <cstdint>
#include <string>

namespace inference {

class QnnBackend {
public:
    QnnBackend() = default;

    ~QnnBackend();

    QnnBackend(const QnnBackend&) = delete;
    QnnBackend& operator=(const QnnBackend&) = delete;

    bool loadLibrary(
        const std::string& backendPath
    );

    bool loadProviders();

    bool selectInterface();

    bool createBackend();

    void shutdown();

    uint32_t providerCount() const noexcept;

    bool interfaceReady() const noexcept;

    bool backendReady() const noexcept;

    Qnn_BackendHandle_t backendHandle() const noexcept;

    QNN_INTERFACE_VER_TYPE& interface() noexcept;

    const std::string& lastError() const noexcept;

private:
    SharedLibrary backendLibrary_;

    const QnnInterface_t** providers_ = nullptr;

    uint32_t providerCount_ = 0;

    QNN_INTERFACE_VER_TYPE qnnInterface_{};

    bool interfaceReady_ = false;

    Qnn_BackendHandle_t backendHandle_ = nullptr;

    std::string lastError_;
};

} // namespace inference