#pragma once

#include "inference/qnn_backend.hpp"
#include "inference/qnn_context.hpp"
#include "inference/shared_library.hpp"

#include <QnnContext.h>

#include <string>

namespace inference {

class QnnModel {
public:
    explicit QnnModel(
        QnnBackend& backend
    ) noexcept;

    ~QnnModel();

    QnnModel(const QnnModel&) = delete;
    QnnModel& operator=(const QnnModel&) = delete;

    bool load(
        const std::string& modelPath
    );

    void shutdown();

    bool ready() const noexcept;

    bool libraryReady() const noexcept;

    bool symbolsReady() const noexcept;

    Qnn_ContextHandle_t contextHandle() const noexcept;

    const std::string& modelPath() const noexcept;

    const std::string& lastError() const noexcept;

private:
    QnnContext context_;

    SharedLibrary modelLibrary_;

    void* composeGraphsSymbol_ = nullptr;

    void* freeGraphsInfoSymbol_ = nullptr;

    std::string modelPath_;

    std::string lastError_;
};

} // namespace inference