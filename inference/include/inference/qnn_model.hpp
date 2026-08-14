#pragma once

#include "inference/qnn_backend.hpp"
#include "inference/qnn_context.hpp"
#include "inference/shared_library.hpp"

#include <QnnLog.h>
#include <QnnWrapperUtils.hpp>

#include <cstdint>
#include <string>
#include <vector>

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

    bool composeGraphs();

    bool finalizeGraphs();

    void shutdown();

    bool ready() const noexcept;

    bool libraryReady() const noexcept;

    bool symbolsReady() const noexcept;

    bool graphsReady() const noexcept;

    bool graphsFinalized() const noexcept;

    bool graphFinalized(
        uint32_t index
    ) const noexcept;

    uint32_t graphCount() const noexcept;

    const qnn_wrapper_api::GraphInfo_t*
    graphInfo(
        uint32_t index
    ) const noexcept;

    Qnn_ContextHandle_t contextHandle() const noexcept;

    const std::string& modelPath() const noexcept;

    const std::string& lastError() const noexcept;

private:
    using ComposeGraphsFn =
        qnn_wrapper_api::ModelError_t (*)(
            Qnn_BackendHandle_t,
            QNN_INTERFACE_VER_TYPE,
            Qnn_ContextHandle_t,
            const qnn_wrapper_api::GraphConfigInfo_t**,
            const uint32_t,
            qnn_wrapper_api::GraphInfo_t***,
            uint32_t*,
            bool,
            QnnLog_Callback_t,
            QnnLog_Level_t
        );

    using FreeGraphsInfoFn =
        qnn_wrapper_api::ModelError_t (*)(
            qnn_wrapper_api::GraphInfo_t***,
            uint32_t
        );

    void releaseGraphs();

private:
    QnnBackend& backend_;

    QnnContext context_;

    SharedLibrary modelLibrary_;

    ComposeGraphsFn composeGraphsFn_ = nullptr;

    FreeGraphsInfoFn freeGraphsInfoFn_ = nullptr;

    qnn_wrapper_api::GraphInfo_t** graphsInfo_ = nullptr;

    uint32_t graphCount_ = 0;

    std::vector<uint8_t> graphFinalized_;

    std::string modelPath_;

    std::string lastError_;
};

} // namespace inference