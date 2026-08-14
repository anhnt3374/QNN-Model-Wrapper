#include "inference/qnn_backend.hpp"
#include "inference/qnn_model.hpp"

#include <QnnTensor.h>
#include <QnnTypes.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace {

void printDimensions(
    const uint32_t* dimensions,
    uint32_t rank
)
{
    std::cout << "[";

    for (uint32_t i = 0; i < rank; ++i) {
        if (i > 0) {
            std::cout << ", ";
        }

        if (dimensions != nullptr) {
            std::cout << dimensions[i];
        } else {
            std::cout << "?";
        }
    }

    std::cout << "]";
}

void printDynamicDimensions(
    const uint8_t* dynamicDimensions,
    uint32_t rank
)
{
    if (dynamicDimensions == nullptr) {
        std::cout << "<all static>";
        return;
    }

    std::cout << "[";

    for (uint32_t i = 0; i < rank; ++i) {
        if (i > 0) {
            std::cout << ", ";
        }

        std::cout
            << static_cast<int>(
                   dynamicDimensions[i]
               );
    }

    std::cout << "]";
}

void printTensorV1(
    const Qnn_TensorV1_t& tensor
)
{
    std::cout
        << "       name: "
        << (
            tensor.name != nullptr
                ? tensor.name
                : "<null>"
        )
        << '\n';

    std::cout
        << "       id: "
        << tensor.id
        << '\n';

    std::cout
        << "       type: "
        << static_cast<int>(tensor.type)
        << '\n';

    std::cout
        << "       data format: "
        << static_cast<int>(tensor.dataFormat)
        << '\n';

    std::cout
        << "       data type: "
        << static_cast<int>(tensor.dataType)
        << '\n';

    std::cout
        << "       rank: "
        << tensor.rank
        << '\n';

    std::cout
        << "       dimensions: ";

    printDimensions(
        tensor.dimensions,
        tensor.rank
    );

    std::cout << '\n';

    std::cout
        << "       mem type: "
        << static_cast<int>(tensor.memType)
        << '\n';

    std::cout
        << "       quantization definition: "
        << static_cast<int>(
               tensor.quantizeParams.encodingDefinition
           )
        << '\n';

    std::cout
        << "       quantization encoding: "
        << static_cast<int>(
               tensor.quantizeParams.quantizationEncoding
           )
        << '\n';
}

void printTensorV2(
    const Qnn_TensorV2_t& tensor
)
{
    std::cout
        << "       name: "
        << (
            tensor.name != nullptr
                ? tensor.name
                : "<null>"
        )
        << '\n';

    std::cout
        << "       id: "
        << tensor.id
        << '\n';

    std::cout
        << "       type: "
        << static_cast<int>(tensor.type)
        << '\n';

    std::cout
        << "       data format: "
        << static_cast<int>(tensor.dataFormat)
        << '\n';

    std::cout
        << "       data type: "
        << static_cast<int>(tensor.dataType)
        << '\n';

    std::cout
        << "       rank: "
        << tensor.rank
        << '\n';

    std::cout
        << "       dimensions: ";

    printDimensions(
        tensor.dimensions,
        tensor.rank
    );

    std::cout << '\n';

    std::cout
        << "       dynamic dimensions: ";

    printDynamicDimensions(
        tensor.isDynamicDimensions,
        tensor.rank
    );

    std::cout << '\n';

    std::cout
        << "       mem type: "
        << static_cast<int>(tensor.memType)
        << '\n';

    std::cout
        << "       quantization definition: "
        << static_cast<int>(
               tensor.quantizeParams.encodingDefinition
           )
        << '\n';

    std::cout
        << "       quantization encoding: "
        << static_cast<int>(
               tensor.quantizeParams.quantizationEncoding
           )
        << '\n';

    std::cout
        << "       is produced: "
        << static_cast<int>(
               tensor.isProduced
           )
        << '\n';
}

void printTensor(
    const Qnn_Tensor_t& tensor,
    const char* prefix,
    uint32_t index
)
{
    std::cout
        << "[INFO] "
        << prefix
        << "["
        << index
        << "]\n";

    std::cout
        << "       version: "
        << static_cast<int>(tensor.version)
        << '\n';

    switch (tensor.version) {

    case QNN_TENSOR_VERSION_1:

        printTensorV1(
            tensor.v1
        );

        break;

    case QNN_TENSOR_VERSION_2:

        printTensorV2(
            tensor.v2
        );

        break;

    default:

        std::cout
            << "       [WARN] Unsupported tensor version\n";

        break;
    }
}

void printGraph(
    const qnn_wrapper_api::GraphInfo_t& graph,
    uint32_t graphIndex
)
{
    std::cout
        << "\n"
        << "========================================\n";

    std::cout
        << "[INFO] graph["
        << graphIndex
        << "]\n";

    std::cout
        << "[INFO] graph name: "
        << (
            graph.graphName != nullptr
                ? graph.graphName
                : "<null>"
        )
        << '\n';

    std::cout
        << "[INFO] graph handle: "
        << graph.graph
        << '\n';

    // =====================================================
    // Inputs
    // =====================================================

    std::cout
        << "[INFO] input count: "
        << graph.numInputTensors
        << '\n';

    if (graph.inputTensors == nullptr &&
        graph.numInputTensors > 0) {

        std::cout
            << "[ERROR] input tensor array is null\n";

        return;
    }

    for (uint32_t i = 0;
         i < graph.numInputTensors;
         ++i) {

        printTensor(
            graph.inputTensors[i],
            "input",
            i
        );
    }

    // =====================================================
    // Outputs
    // =====================================================

    std::cout
        << "[INFO] output count: "
        << graph.numOutputTensors
        << '\n';

    if (graph.outputTensors == nullptr &&
        graph.numOutputTensors > 0) {

        std::cout
            << "[ERROR] output tensor array is null\n";

        return;
    }

    for (uint32_t i = 0;
         i < graph.numOutputTensors;
         ++i) {

        printTensor(
            graph.outputTensors[i],
            "output",
            i
        );
    }
}

} // namespace

int main()
{
    const char* backendPath =
        std::getenv("QNN_BACKEND_PATH");

    const char* modelPath =
        std::getenv("QNN_MODEL_PATH");

    if (backendPath == nullptr) {
        std::cerr
            << "[ERROR] QNN_BACKEND_PATH is not set\n";

        return 1;
    }

    if (modelPath == nullptr) {
        std::cerr
            << "[ERROR] QNN_MODEL_PATH is not set\n";

        return 1;
    }

    std::cout
        << "[INFO] backend: "
        << backendPath
        << '\n';

    std::cout
        << "[INFO] model: "
        << modelPath
        << '\n';

    // =====================================================
    // QNN runtime
    // =====================================================

    inference::QnnBackend backend;

    if (!backend.loadLibrary(backendPath)) {
        std::cerr
            << "[ERROR] loadLibrary: "
            << backend.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] QNN backend library loaded\n";

    if (!backend.loadProviders()) {
        std::cerr
            << "[ERROR] loadProviders: "
            << backend.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] QNN providers loaded\n";

    if (!backend.selectInterface()) {
        std::cerr
            << "[ERROR] selectInterface: "
            << backend.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] QNN interface selected\n";

    if (!backend.createBackend()) {
        std::cerr
            << "[ERROR] createBackend: "
            << backend.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] QNN backend created\n";

    if (!backend.createDevice()) {
        std::cerr
            << "[ERROR] createDevice: "
            << backend.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] QNN device created\n";

    // =====================================================
    // Model
    // =====================================================

    inference::QnnModel model(backend);

    if (!model.load(modelPath)) {
        std::cerr
            << "[ERROR] model.load: "
            << model.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] model .so loaded\n";

    std::cout
        << "[PASS] required model symbols found\n";

    // =====================================================
    // Compose graph
    // =====================================================

    if (!model.composeGraphs()) {
        std::cerr
            << "[ERROR] composeGraphs: "
            << model.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] QNN graphs composed\n";

    std::cout
        << "[INFO] graph count: "
        << model.graphCount()
        << '\n';

    // =====================================================
    // Inspect graphs
    // =====================================================

    for (uint32_t graphIndex = 0;
         graphIndex < model.graphCount();
         ++graphIndex) {

        const auto* graph =
            model.graphInfo(
                graphIndex
            );

        if (graph == nullptr) {
            std::cerr
                << "[ERROR] graphInfo("
                << graphIndex
                << ") returned null\n";

            return 1;
        }

        printGraph(
            *graph,
            graphIndex
        );
    }

    std::cout
        << "\n"
        << "[PASS] graph metadata inspection complete\n";

    return 0;
}