#include "inference/qnn_model.hpp"

namespace inference {

QnnModel::QnnModel(
    QnnBackend& backend
) noexcept
    : context_(backend)
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
    // 1. Create an independent QNN context for this model
    // =====================================================

    if (!context_.create()) {
        lastError_ =
            "Failed to create QNN context: "
            + context_.lastError();

        return false;
    }

    // =====================================================
    // 2. Load model shared library
    // =====================================================

    if (!modelLibrary_.open(modelPath)) {
        lastError_ =
            "Failed to load model library: "
            + modelLibrary_.lastError();

        context_.shutdown();

        return false;
    }

    // =====================================================
    // 3. Find QnnModel_composeGraphs
    // =====================================================

    composeGraphsSymbol_ =
        modelLibrary_.getSymbol<void*>(
            "QnnModel_composeGraphs"
        );

    if (composeGraphsSymbol_ == nullptr) {
        lastError_ =
            "Model library does not export "
            "QnnModel_composeGraphs: "
            + modelLibrary_.lastError();

        shutdown();

        return false;
    }

    // =====================================================
    // 4. Find QnnModel_freeGraphsInfo
    // =====================================================

    freeGraphsInfoSymbol_ =
        modelLibrary_.getSymbol<void*>(
            "QnnModel_freeGraphsInfo"
        );

    if (freeGraphsInfoSymbol_ == nullptr) {
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

void QnnModel::shutdown()
{
    composeGraphsSymbol_ = nullptr;

    freeGraphsInfoSymbol_ = nullptr;

    modelLibrary_.close();

    context_.shutdown();

    modelPath_.clear();
}

bool QnnModel::ready() const noexcept
{
    return
        context_.ready()
        &&
        modelLibrary_.isOpen()
        &&
        composeGraphsSymbol_ != nullptr
        &&
        freeGraphsInfoSymbol_ != nullptr;
}

bool QnnModel::libraryReady() const noexcept
{
    return modelLibrary_.isOpen();
}

bool QnnModel::symbolsReady() const noexcept
{
    return
        composeGraphsSymbol_ != nullptr
        &&
        freeGraphsInfoSymbol_ != nullptr;
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