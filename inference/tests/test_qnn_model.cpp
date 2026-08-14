#include "inference/qnn_backend.hpp"
#include "inference/qnn_model.hpp"

#include <QnnTensor.h>
#include <QnnTypes.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace {

const char* dataTypeName(
    Qnn_DataType_t dataType
)
{
    switch (dataType) {

    case QNN_DATATYPE_FLOAT_32:
        return "FLOAT_32";

    case QNN_DATATYPE_FLOAT_16:
        return "FLOAT_16";

    case QNN_DATATYPE_INT_8:
        return "INT_8";

    case QNN_DATATYPE_INT_16:
        return "INT_16";

    case QNN_DATATYPE_UINT_8:
        return "UINT_8";

    case QNN_DATATYPE_UINT_16:
        return "UINT_16";

    case QNN_DATATYPE_SFIXED_POINT_8:
        return "SFIXED_POINT_8";

    case QNN_DATATYPE_SFIXED_POINT_16:
        return "SFIXED_POINT_16";

    case QNN_DATATYPE_UFIXED_POINT_8:
        return "UFIXED_POINT_8";

    case QNN_DATATYPE_UFIXED_POINT_16:
        return "UFIXED_POINT_16";

    default:
        return "UNKNOWN";
    }
}

uint32_t bytesPerElement(
    Qnn_DataType_t dataType
)
{
    switch (dataType) {

    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
    case QNN_DATATYPE_BOOL_8:
        return 1;

    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
        return 2;

    case QNN_DATATYPE_FLOAT_32:
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_UINT_32:
    case QNN_DATATYPE_SFIXED_POINT_32:
    case QNN_DATATYPE_UFIXED_POINT_32:
        return 4;

    default:
        return 0;
    }
}

uint64_t elementCount(
    const uint32_t* dimensions,
    uint32_t rank
)
{
    if (dimensions == nullptr ||
        rank == 0) {

        return 0;
    }

    uint64_t count = 1;

    for (uint32_t i = 0;
         i < rank;
         ++i) {

        count *= dimensions[i];
    }

    return count;
}

void printDimensions(
    const uint32_t* dimensions,
    uint32_t rank
)
{
    std::cout << "[";

    for (uint32_t i = 0;
         i < rank;
         ++i) {

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

    for (uint32_t i = 0;
         i < rank;
         ++i) {

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

void printQuantization(
    const Qnn_QuantizeParams_t& quantizeParams
)
{
    std::cout
        << "       quantization definition: "
        << static_cast<int>(
               quantizeParams.encodingDefinition
           )
        << '\n';

    std::cout
        << "       quantization encoding: "
        << static_cast<int>(
               quantizeParams.quantizationEncoding
           )
        << '\n';

    if (quantizeParams.quantizationEncoding ==
        QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {

        std::cout
            << "       quantization type: "
            << "SCALE_OFFSET\n";

        std::cout
            << "       scale: "
            << quantizeParams
                   .scaleOffsetEncoding
                   .scale
            << '\n';

        std::cout
            << "       offset: "
            << quantizeParams
                   .scaleOffsetEncoding
                   .offset
            << '\n';
    }
}

void printTensorCommon(
    const char* name,
    uint32_t id,
    Qnn_TensorType_t type,
    Qnn_TensorDataFormat_t dataFormat,
    Qnn_DataType_t dataType,
    const Qnn_QuantizeParams_t& quantizeParams,
    uint32_t rank,
    const uint32_t* dimensions,
    Qnn_TensorMemType_t memType
)
{
    std::cout
        << "       name: "
        << (
            name != nullptr
                ? name
                : "<null>"
        )
        << '\n';

    std::cout
        << "       id: "
        << id
        << '\n';

    std::cout
        << "       type: "
        << static_cast<int>(type)
        << '\n';

    std::cout
        << "       data format: "
        << dataFormat
        << '\n';

    std::cout
        << "       data type: "
        << static_cast<int>(dataType)
        << " ("
        << dataTypeName(dataType)
        << ")"
        << '\n';

    std::cout
        << "       rank: "
        << rank
        << '\n';

    std::cout
        << "       dimensions: ";

    printDimensions(
        dimensions,
        rank
    );

    std::cout << '\n';

    const uint64_t elements =
        elementCount(
            dimensions,
            rank
        );

    const uint32_t elementBytes =
        bytesPerElement(
            dataType
        );

    const uint64_t bufferBytes =
        elements *
        static_cast<uint64_t>(
            elementBytes
        );

    std::cout
        << "       element count: "
        << elements
        << '\n';

    std::cout
        << "       bytes / element: "
        << elementBytes
        << '\n';

    std::cout
        << "       buffer bytes: "
        << bufferBytes
        << '\n';

    std::cout
        << "       mem type: "
        << static_cast<int>(memType)
        << '\n';

    printQuantization(
        quantizeParams
    );
}

void printTensorV1(
    const Qnn_TensorV1_t& tensor
)
{
    printTensorCommon(
        tensor.name,
        tensor.id,
        tensor.type,
        tensor.dataFormat,
        tensor.dataType,
        tensor.quantizeParams,
        tensor.rank,
        tensor.dimensions,
        tensor.memType
    );
}

void printTensorV2(
    const Qnn_TensorV2_t& tensor
)
{
    printTensorCommon(
        tensor.name,
        tensor.id,
        tensor.type,
        tensor.dataFormat,
        tensor.dataType,
        tensor.quantizeParams,
        tensor.rank,
        tensor.dimensions,
        tensor.memType
    );

    std::cout
        << "       dynamic dimensions: ";

    printDynamicDimensions(
        tensor.isDynamicDimensions,
        tensor.rank
    );

    std::cout << '\n';
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
        << static_cast<int>(
               tensor.version
           )
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
        std::getenv(
            "QNN_BACKEND_PATH"
        );

    const char* modelPath =
        std::getenv(
            "QNN_MODEL_PATH"
        );

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
    // Runtime
    // =====================================================

    inference::QnnBackend backend;

    if (!backend.loadLibrary(
            backendPath
        )) {

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

    inference::QnnModel model(
        backend
    );

    if (!model.load(
            modelPath
        )) {

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
    // Metadata inspection
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