#include "inference/qnn_model.hpp"

#include <QnnGraph.h>

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
    // 1. Create context
    // =====================================================

    if (!context_.create()) {
        lastError_ =
            "Failed to create QNN context: "
            + context_.lastError();

        return false;
    }

    // =====================================================
    // 2. Load generated model library
    // =====================================================

    if (!modelLibrary_.open(modelPath)) {
        lastError_ =
            "Failed to load model library: "
            + modelLibrary_.lastError();

        context_.shutdown();

        return false;
    }

    // =====================================================
    // 3. Resolve compose function
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
    // 4. Resolve graph metadata free function
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
    if (!ready()) {
        lastError_ =
            "Model is not ready";

        return false;
    }

    if (graphsInfo_ != nullptr &&
        graphCount_ > 0) {

        return true;
    }

    graphsInfo_ = nullptr;
    graphCount_ = 0;

    graphFinalized_.clear();

    const qnn_wrapper_api::ModelError_t result =
        composeGraphsFn_(
            backend_.backendHandle(),
            backend_.interface(),
            context_.handle(),

            nullptr,
            0,

            &graphsInfo_,
            &graphCount_,

            false,

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

    graphFinalized_.assign(
        graphCount_,
        0
    );

    lastError_.clear();

    return true;
}

bool QnnModel::finalizeGraphs()
{
    if (!graphsReady()) {
        lastError_ =
            "Graphs are not ready";

        return false;
    }

    if (!backend_.interfaceReady()) {
        lastError_ =
            "QNN interface is not ready";

        return false;
    }

    if (graphFinalized_.size() !=
        graphCount_) {

        graphFinalized_.assign(
            graphCount_,
            0
        );
    }

    for (uint32_t i = 0;
         i < graphCount_;
         ++i) {

        if (graphFinalized_[i] != 0) {
            continue;
        }

        if (graphsInfo_[i] == nullptr) {
            std::ostringstream oss;

            oss
                << "GraphInfo is null at index "
                << i;

            lastError_ = oss.str();

            return false;
        }

        if (graphsInfo_[i]->graph == nullptr) {
            std::ostringstream oss;

            oss
                << "Graph handle is null at index "
                << i;

            lastError_ = oss.str();

            return false;
        }

        const Qnn_ErrorHandle_t result =
            backend_.interface().graphFinalize(
                graphsInfo_[i]->graph,
                nullptr,
                nullptr
            );

        if (result != QNN_SUCCESS) {
            std::ostringstream oss;

            oss
                << "graphFinalize failed for graph["
                << i
                << "]";

            if (graphsInfo_[i]->graphName != nullptr) {
                oss
                    << " ("
                    << graphsInfo_[i]->graphName
                    << ")";
            }

            oss
                << ". error="
                << result;

            lastError_ = oss.str();

            return false;
        }

        graphFinalized_[i] = 1;
    }

    lastError_.clear();

    return true;
}

bool QnnModel::executeGraph(
    uint32_t graphIndex,
    const Qnn_Tensor_t* inputTensors,
    uint32_t inputCount,
    Qnn_Tensor_t* outputTensors,
    uint32_t outputCount
)
{
    // =====================================================
    // Validate graph
    // =====================================================

    if (!graphsReady()) {
        lastError_ =
            "Graphs are not ready";

        return false;
    }

    if (graphIndex >= graphCount_) {
        std::ostringstream oss;

        oss
            << "Invalid graph index: "
            << graphIndex;

        lastError_ = oss.str();

        return false;
    }

    if (!graphFinalized(
            graphIndex
        )) {

        std::ostringstream oss;

        oss
            << "Graph["
            << graphIndex
            << "] is not finalized";

        lastError_ = oss.str();

        return false;
    }

    const auto* graphInfo =
        graphsInfo_[graphIndex];

    if (graphInfo == nullptr) {
        lastError_ =
            "GraphInfo is null";

        return false;
    }

    if (graphInfo->graph == nullptr) {
        lastError_ =
            "Graph handle is null";

        return false;
    }

    // =====================================================
    // Validate input tensors
    // =====================================================

    if (inputCount !=
        graphInfo->numInputTensors) {

        std::ostringstream oss;

        oss
            << "Input tensor count mismatch. expected="
            << graphInfo->numInputTensors
            << ", actual="
            << inputCount;

        lastError_ = oss.str();

        return false;
    }

    if (inputCount > 0 &&
        inputTensors == nullptr) {

        lastError_ =
            "Input tensor array is null";

        return false;
    }

    // =====================================================
    // Validate output tensors
    // =====================================================

    if (outputCount !=
        graphInfo->numOutputTensors) {

        std::ostringstream oss;

        oss
            << "Output tensor count mismatch. expected="
            << graphInfo->numOutputTensors
            << ", actual="
            << outputCount;

        lastError_ = oss.str();

        return false;
    }

    if (outputCount > 0 &&
        outputTensors == nullptr) {

        lastError_ =
            "Output tensor array is null";

        return false;
    }

    // =====================================================
    // Execute graph
    // =====================================================

    const Qnn_ErrorHandle_t result =
        backend_.interface().graphExecute(
            graphInfo->graph,

            inputTensors,
            inputCount,

            outputTensors,
            outputCount,

            // No profiling yet.
            nullptr,

            // No timeout/cancellation signal yet.
            nullptr
        );

    if (result != QNN_SUCCESS) {
        std::ostringstream oss;

        oss
            << "graphExecute failed for graph["
            << graphIndex
            << "]";

        if (graphInfo->graphName != nullptr) {
            oss
                << " ("
                << graphInfo->graphName
                << ")";
        }

        oss
            << ". error="
            << result;

        lastError_ = oss.str();

        return false;
    }

    lastError_.clear();

    return true;
}

void QnnModel::releaseGraphs()
{
    graphFinalized_.clear();

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

bool QnnModel::graphsFinalized() const noexcept
{
    if (!graphsReady()) {
        return false;
    }

    if (graphFinalized_.size() !=
        graphCount_) {

        return false;
    }

    for (uint8_t finalized :
         graphFinalized_) {

        if (finalized == 0) {
            return false;
        }
    }

    return true;
}

bool QnnModel::graphFinalized(
    uint32_t index
) const noexcept
{
    if (index >=
        graphFinalized_.size()) {

        return false;
    }

    return
        graphFinalized_[index] != 0;
}

uint32_t
QnnModel::graphCount() const noexcept
{
    return graphCount_;
}

const qnn_wrapper_api::GraphInfo_t*
QnnModel::graphInfo(
    uint32_t index
) const noexcept
{
    if (graphsInfo_ == nullptr) {
        return nullptr;
    }

    if (index >= graphCount_) {
        return nullptr;
    }

    return graphsInfo_[index];
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