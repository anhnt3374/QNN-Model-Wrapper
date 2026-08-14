#include "inference/qnn_backend.hpp"
#include "inference/qnn_model.hpp"
#include "inference/qnn_quantization.hpp"
#include "inference/qnn_tensor_buffer.hpp"

#include <QnnTypes.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

namespace {

using TensorBufferPtr =
    std::unique_ptr<
        inference::QnnTensorBuffer
    >;

// =========================================================
// Tensor buffer creation
// =========================================================

bool createTensorBuffers(
    const Qnn_Tensor_t* tensors,
    uint32_t tensorCount,
    std::vector<TensorBufferPtr>& buffers,
    const char* prefix
)
{
    if (tensorCount > 0 &&
        tensors == nullptr) {

        std::cerr
            << "[ERROR] "
            << prefix
            << " tensor metadata is null\n";

        return false;
    }

    buffers.reserve(
        tensorCount
    );

    for (uint32_t i = 0;
         i < tensorCount;
         ++i) {

        auto buffer =
            std::make_unique<
                inference::QnnTensorBuffer
            >();

        if (!buffer->initialize(
                tensors[i]
            )) {

            std::cerr
                << "[ERROR] failed to initialize "
                << prefix
                << "["
                << i
                << "]: "
                << buffer->lastError()
                << '\n';

            return false;
        }

        std::cout
            << "[PASS] "
            << prefix
            << "["
            << i
            << "] buffer allocated\n";

        std::cout
            << "       name: "
            << buffer->name()
            << '\n';

        std::cout
            << "       buffer bytes: "
            << buffer->byteSize()
            << '\n';

        buffers.push_back(
            std::move(buffer)
        );
    }

    return true;
}

// =========================================================
// Build contiguous Qnn_Tensor_t arrays
// =========================================================

std::vector<Qnn_Tensor_t>
buildTensorArray(
    const std::vector<TensorBufferPtr>& buffers
)
{
    std::vector<Qnn_Tensor_t>
        tensors;

    tensors.reserve(
        buffers.size()
    );

    for (const auto& buffer :
         buffers) {

        tensors.push_back(
            buffer->tensor()
        );
    }

    return tensors;
}

// =========================================================
// Generic tensor metadata access
// =========================================================

Qnn_DataType_t getDataType(
    const Qnn_Tensor_t& tensor
)
{
    switch (tensor.version) {

    case QNN_TENSOR_VERSION_1:
        return tensor.v1.dataType;

    case QNN_TENSOR_VERSION_2:
        return tensor.v2.dataType;

    default:
        return QNN_DATATYPE_UNDEFINED;
    }
}

const Qnn_QuantizeParams_t*
getQuantization(
    const Qnn_Tensor_t& tensor
)
{
    switch (tensor.version) {

    case QNN_TENSOR_VERSION_1:
        return &tensor.v1.quantizeParams;

    case QNN_TENSOR_VERSION_2:
        return &tensor.v2.quantizeParams;

    default:
        return nullptr;
    }
}

// =========================================================
// SCRFD neutral input
// =========================================================

bool fillFaceDetectorTestInput(
    inference::QnnTensorBuffer& buffer
)
{
    const Qnn_Tensor_t& tensor =
        buffer.tensor();

    if (getDataType(
            tensor
        ) !=
        QNN_DATATYPE_UFIXED_POINT_16) {

        std::cerr
            << "[ERROR] SCRFD test expects "
            << "UFIXED_POINT_16 input\n";

        return false;
    }

    const Qnn_QuantizeParams_t* quantization =
        getQuantization(
            tensor
        );

    if (quantization == nullptr) {

        std::cerr
            << "[ERROR] SCRFD input quantization "
            << "metadata missing\n";

        return false;
    }

    if (quantization->quantizationEncoding !=
        QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {

        std::cerr
            << "[ERROR] SCRFD test expects "
            << "SCALE_OFFSET quantization\n";

        return false;
    }

    const int64_t quantizedZero =
        -static_cast<int64_t>(
            quantization
                ->scaleOffsetEncoding
                .offset
        );

    if (quantizedZero < 0 ||
        quantizedZero >
            std::numeric_limits<uint16_t>::max()) {

        std::cerr
            << "[ERROR] quantized zero outside "
            << "uint16 range: "
            << quantizedZero
            << '\n';

        return false;
    }

    auto* data =
        static_cast<uint16_t*>(
            buffer.data()
        );

    if (data == nullptr) {

        std::cerr
            << "[ERROR] SCRFD input buffer is null\n";

        return false;
    }

    std::fill_n(
        data,
        static_cast<std::size_t>(
            buffer.elementCount()
        ),
        static_cast<uint16_t>(
            quantizedZero
        )
    );

    std::cout
        << "[INFO] SCRFD test input filled\n";

    std::cout
        << "       quantized zero: "
        << quantizedZero
        << '\n';

    std::cout
        << "       scale: "
        << quantization
               ->scaleOffsetEncoding
               .scale
        << '\n';

    std::cout
        << "       offset: "
        << quantization
               ->scaleOffsetEncoding
               .offset
        << '\n';

    std::cout
        << "       elements: "
        << buffer.elementCount()
        << '\n';

    return true;
}

// =========================================================
// Clear output memory before graphExecute
// =========================================================

void clearOutputBuffers(
    std::vector<TensorBufferPtr>& buffers
)
{
    for (auto& buffer :
         buffers) {

        if (buffer->data() == nullptr) {
            continue;
        }

        std::memset(
            buffer->data(),
            0,
            buffer->byteSize()
        );
    }
}

// =========================================================
// Print raw + dequantized output
//
// 5G.2 scope:
//
// uint16_t
//      ↓
// scale-offset dequantization
//      ↓
// float
//
// No SCRFD bbox/keypoint decoding yet.
// =========================================================

bool printDequantizedOutputSample(
    const inference::QnnTensorBuffer& buffer,
    std::size_t maxValues = 8
)
{
    const Qnn_Tensor_t& tensor =
        buffer.tensor();

    if (getDataType(
            tensor
        ) !=
        QNN_DATATYPE_UFIXED_POINT_16) {

        std::cerr
            << "[ERROR] "
            << buffer.name()
            << " is not UFIXED_POINT_16\n";

        return false;
    }

    const Qnn_QuantizeParams_t* quantization =
        getQuantization(
            tensor
        );

    if (quantization == nullptr) {

        std::cerr
            << "[ERROR] "
            << buffer.name()
            << " has no quantization metadata\n";

        return false;
    }

    if (quantization->quantizationEncoding !=
        QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {

        std::cerr
            << "[ERROR] "
            << buffer.name()
            << " does not use SCALE_OFFSET\n";

        return false;
    }

    const auto* rawValues =
        static_cast<const uint16_t*>(
            buffer.data()
        );

    if (rawValues == nullptr) {

        std::cerr
            << "[ERROR] "
            << buffer.name()
            << " data buffer is null\n";

        return false;
    }

    const float scale =
        quantization
            ->scaleOffsetEncoding
            .scale;

    const int32_t offset =
        quantization
            ->scaleOffsetEncoding
            .offset;

    const std::size_t count =
        std::min<std::size_t>(
            maxValues,
            static_cast<std::size_t>(
                buffer.elementCount()
            )
        );

    // -----------------------------------------------------
    // Quantization metadata
    // -----------------------------------------------------

    std::cout
        << "       scale: "
        << scale
        << '\n';

    std::cout
        << "       offset: "
        << offset
        << '\n';

    // -----------------------------------------------------
    // Raw uint16 values
    // -----------------------------------------------------

    std::cout
        << "       raw: [";

    for (std::size_t i = 0;
         i < count;
         ++i) {

        if (i > 0) {
            std::cout << ", ";
        }

        std::cout
            << rawValues[i];
    }

    std::cout << "]\n";

    // -----------------------------------------------------
    // Dequantized float values
    // -----------------------------------------------------

    std::cout
        << "       float: [";

    std::cout
        << std::fixed
        << std::setprecision(6);

    for (std::size_t i = 0;
         i < count;
         ++i) {

        if (i > 0) {
            std::cout << ", ";
        }

        const float value =
            inference::dequantizeScaleOffset(
                rawValues[i],
                scale,
                offset
            );

        std::cout
            << value;
    }

    std::cout
        << "]\n"
        << std::defaultfloat;

    return true;
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
    // QNN backend
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
    // SCRFD model
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
        << "[PASS] SCRFD model loaded\n";

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
        << "[PASS] SCRFD graph composed\n";

    // =====================================================
    // Finalize graph
    // =====================================================

    if (!model.finalizeGraphs()) {

        std::cerr
            << "[ERROR] finalizeGraphs: "
            << model.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] SCRFD graph finalized\n";

    // =====================================================
    // SCRFD graph 0
    // =====================================================

    const auto* graph =
        model.graphInfo(
            0
        );

    if (graph == nullptr) {

        std::cerr
            << "[ERROR] graphInfo(0) returned null\n";

        return 1;
    }

    std::cout
        << "[INFO] graph name: "
        << (
            graph->graphName != nullptr
                ? graph->graphName
                : "<null>"
        )
        << '\n';

    std::cout
        << "[INFO] inputs: "
        << graph->numInputTensors
        << '\n';

    std::cout
        << "[INFO] outputs: "
        << graph->numOutputTensors
        << '\n';

    // =====================================================
    // Allocate runtime tensor buffers
    // =====================================================

    std::vector<TensorBufferPtr>
        inputBuffers;

    std::vector<TensorBufferPtr>
        outputBuffers;

    if (!createTensorBuffers(
            graph->inputTensors,
            graph->numInputTensors,
            inputBuffers,
            "input"
        )) {

        return 1;
    }

    if (!createTensorBuffers(
            graph->outputTensors,
            graph->numOutputTensors,
            outputBuffers,
            "output"
        )) {

        return 1;
    }

    std::cout
        << "[PASS] SCRFD runtime buffers ready\n";

    // =====================================================
    // Validate SCRFD input count
    // =====================================================

    if (inputBuffers.size() != 1) {

        std::cerr
            << "[ERROR] SCRFD expected 1 input, got "
            << inputBuffers.size()
            << '\n';

        return 1;
    }

    // =====================================================
    // Fill neutral input
    // =====================================================

    if (!fillFaceDetectorTestInput(
            *inputBuffers[0]
        )) {

        return 1;
    }

    // =====================================================
    // Clear output memory
    // =====================================================

    clearOutputBuffers(
        outputBuffers
    );

    // =====================================================
    // Build contiguous Qnn_Tensor_t arrays
    // =====================================================

    std::vector<Qnn_Tensor_t>
        inputTensors =
            buildTensorArray(
                inputBuffers
            );

    std::vector<Qnn_Tensor_t>
        outputTensors =
            buildTensorArray(
                outputBuffers
            );

    // =====================================================
    // Execute SCRFD on HTP
    // =====================================================

    std::cout
        << "[INFO] executing SCRFD graph on HTP...\n";

    if (!model.executeGraph(
            0,

            inputTensors.data(),
            static_cast<uint32_t>(
                inputTensors.size()
            ),

            outputTensors.data(),
            static_cast<uint32_t>(
                outputTensors.size()
            )
        )) {

        std::cerr
            << "[ERROR] executeGraph: "
            << model.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] SCRFD graphExecute succeeded\n";

    // =====================================================
    // 5G.2
    //
    // Dequantize all SCRFD output samples
    // =====================================================

    std::cout
        << "[INFO] dequantized SCRFD outputs:\n";

    for (std::size_t i = 0;
         i < outputBuffers.size();
         ++i) {

        std::cout
            << "[INFO] output["
            << i
            << "] "
            << outputBuffers[i]->name()
            << '\n';

        if (!printDequantizedOutputSample(
                *outputBuffers[i]
            )) {

            return 1;
        }
    }

    std::cout
        << "[PASS] 5G.2 SCRFD "
        << "dequantization test complete\n";

    return 0;
}