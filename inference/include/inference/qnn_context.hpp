#pragma once

#include "inference/qnn_backend.hpp"

#include <QnnContext.h>

#include <string>

namespace inference {

class QnnContext {
public:
    explicit QnnContext(
        QnnBackend& backend
    ) noexcept;

    ~QnnContext();

    QnnContext(const QnnContext&) = delete;
    QnnContext& operator=(const QnnContext&) = delete;

    bool create();

    void shutdown();

    bool ready() const noexcept;

    Qnn_ContextHandle_t handle() const noexcept;

    const std::string& lastError() const noexcept;

private:
    QnnBackend& backend_;

    Qnn_ContextHandle_t contextHandle_ = nullptr;

    std::string lastError_;
};

} // namespace inference