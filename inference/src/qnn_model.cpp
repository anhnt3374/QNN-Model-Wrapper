#include "inference/qnn_model.hpp"

#include <sstream>

namespace inference {

QnnModel::QnnModel(
    QnnBackend& backend
) noexcept
    : backend_(backend),
      context_(backend)
{
}

QnnModel::~QnnModel()
{
    shutdown();
}

bool QnnModel::load(
    const std::string& modelPath
)
{
    shutdown();

    // =====================================================
    // 1. Create independent context for this model
    // =====================================================

    if (!context_.create()) {
        lastError_ =
            "Failed to create QNN context: "
            + context_.lastError();

        return false;
    }

    // =====================================================
    // 2. Load generated model .so
    // =====================================================

    if (!modelLibrary_.open(modelPath)) {
        lastError_ =
            "Failed to load model library: "
            + modelLibrary_.lastError();

        context_.shutdown();

        return false;
    }

    // =====================================================
    // 3. Resolve QnnModel_composeGraphs
    // =====================================================

    composeGraphsFn_ =
        modelLibrary_.getSymbol<ComposeGraphsFn>(
            "QnnModel_composeGraphs"
        );

    if (composeGraphsFn_ == nullptr) {
        lastError_ =
            "Model library does not export "
            "QnnModel_composeGraphs: "
            + modelLibrary_.lastError();

        shutdown();

        return false;
    }

    // =====================================================
    // 4. Resolve QnnModel_freeGraphsInfo
    // =====================================================

    freeGraphsInfoFn_ =
        modelLibrary_.getSymbol<FreeGraphsInfoFn>(
            "QnnModel_freeGraphsInfo"
        );

    if (freeGraphsInfoFn_ == nullptr) {
        lastError_ =
            "Model library does not export "
            "QnnModel_freeGraphsInfo: "
            + modelLibrary_.lastError();

        shutdown();

        return false;
    }

    modelPath_ = modelPath;

    lastError_.clear();

    return true;
}

bool QnnModel::composeGraphs()
{
    // =====================================================
    // Validate runtime state
    // =====================================================

    if (!ready()) {
        lastError_ =
            "Model is not ready";

        return false;
    }

    // composeGraphs() is idempotent in our wrapper.
    if (graphsInfo_ != nullptr &&
        graphCount_ > 0) {

        return true;
    }

    graphsInfo_ = nullptr;
    graphCount_ = 0;

    // =====================================================
    // Call generated model function
    // =====================================================

    const qnn_wrapper_api::ModelError_t result =
        composeGraphsFn_(
            backend_.backendHandle(),
            backend_.interface(),
            context_.handle(),

            // No graph-specific config for now.
            nullptr,
            0,

            &graphsInfo_,
            &graphCount_,

            // Debug graph generation disabled.
            false,

            // We have not implemented QNN logger yet.
            nullptr,
            QNN_LOG_LEVEL_ERROR
        );

    if (result !=
        qnn_wrapper_api::MODEL_NO_ERROR) {

        std::ostringstream oss;

        oss
            << "QnnModel_composeGraphs failed. error="
            << static_cast<int>(result);

        lastError_ = oss.str();

        releaseGraphs();

        return false;
    }

    if (graphsInfo_ == nullptr) {
        lastError_ =
            "QnnModel_composeGraphs returned null graphsInfo";

        releaseGraphs();

        return false;
    }

    if (graphCount_ == 0) {
        lastError_ =
            "QnnModel_composeGraphs returned zero graphs";

        releaseGraphs();

        return false;
    }

    lastError_.clear();

    return true;
}

void QnnModel::releaseGraphs()
{
    if (graphsInfo_ == nullptr) {
        graphCount_ = 0;
        return;
    }

    if (freeGraphsInfoFn_ != nullptr) {
        freeGraphsInfoFn_(
            &graphsInfo_,
            graphCount_
        );
    }

    graphsInfo_ = nullptr;

    graphCount_ = 0;
}

void QnnModel::shutdown()
{
    // Important:
    //
    // QnnModel_freeGraphsInfo lives inside model .so,
    // therefore graph metadata must be released BEFORE
    // modelLibrary_.close().

    releaseGraphs();

    composeGraphsFn_ = nullptr;

    freeGraphsInfoFn_ = nullptr;

    modelLibrary_.close();

    context_.shutdown();

    modelPath_.clear();
}

bool QnnModel::ready() const noexcept
{
    return
        backend_.backendReady()
        &&
        backend_.deviceReady()
        &&
        context_.ready()
        &&
        modelLibrary_.isOpen()
        &&
        composeGraphsFn_ != nullptr
        &&
        freeGraphsInfoFn_ != nullptr;
}

bool QnnModel::libraryReady() const noexcept
{
    return modelLibrary_.isOpen();
}

bool QnnModel::symbolsReady() const noexcept
{
    return
        composeGraphsFn_ != nullptr
        &&
        freeGraphsInfoFn_ != nullptr;
}

bool QnnModel::graphsReady() const noexcept
{
    return
        graphsInfo_ != nullptr
        &&
        graphCount_ > 0;
}

uint32_t
QnnModel::graphCount() const noexcept
{
    return graphCount_;
}

Qnn_ContextHandle_t
QnnModel::contextHandle() const noexcept
{
    return context_.handle();
}

const std::string&
QnnModel::modelPath() const noexcept
{
    return modelPath_;
}

const std::string&
QnnModel::lastError() const noexcept
{
    return lastError_;
}

} // namespace inference