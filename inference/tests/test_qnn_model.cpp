#include "inference/qnn_backend.hpp"
#include "inference/qnn_model.hpp"
#include "inference/qnn_tensor_buffer.hpp"

#include <QnnTypes.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
//
// graphExecute expects contiguous arrays of Qnn_Tensor_t.
// QnnTensorBuffer objects themselves are owned separately.
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
// Access tensor metadata
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
// SCRFD test input
//
// Real zero:
//
// float = (q + offset) * scale
//
// 0 = q + offset
//
// q = -offset
//
// SCRFD input:
// offset = -32768
//
// therefore:
// q = 32768
// =========================================================

bool fillFaceDetectorTestInput(
    inference::QnnTensorBuffer& buffer
)
{
    const Qnn_Tensor_t& tensor =
        buffer.tensor();

    const Qnn_DataType_t dataType =
        getDataType(
            tensor
        );

    if (dataType !=
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
            << "[ERROR] SCRFD input "
            << "quantization metadata missing\n";

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
            << "[ERROR] quantized zero is "
            << "outside uint16 range: "
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
        << "       real zero scale: "
        << quantization
               ->scaleOffsetEncoding
               .scale
        << '\n';

    std::cout
        << "       elements: "
        << buffer.elementCount()
        << '\n';

    return true;
}

// =========================================================
// Clear outputs before execution
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
// Print several raw uint16 output values
//
// 5G.1 does NOT interpret detections yet.
// =========================================================

void printRawOutputSample(
    const inference::QnnTensorBuffer& buffer,
    std::size_t maxValues = 8
)
{
    if (buffer.data() == nullptr) {
        return;
    }

    if (getDataType(
            buffer.tensor()
        ) !=
        QNN_DATATYPE_UFIXED_POINT_16) {

        std::cout
            << "       raw sample: "
            << "<unsupported datatype>\n";

        return;
    }

    const auto* values =
        static_cast<const uint16_t*>(
            buffer.data()
        );

    const std::size_t count =
        std::min<std::size_t>(
            maxValues,
            static_cast<std::size_t>(
                buffer.elementCount()
            )
        );

    std::cout
        << "       raw sample: [";

    for (std::size_t i = 0;
         i < count;
         ++i) {

        if (i > 0) {
            std::cout << ", ";
        }

        std::cout
            << values[i];
    }

    std::cout << "]\n";
}

// =========================================================
// Main
// =========================================================

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
    // Compose
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
    // Finalize
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
    // SCRFD uses graph 0
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
    // Allocate runtime buffers
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
    // SCRFD has exactly one input
    // =====================================================

    if (inputBuffers.size() != 1) {

        std::cerr
            << "[ERROR] SCRFD expected 1 input, got "
            << inputBuffers.size()
            << '\n';

        return 1;
    }

    // =====================================================
    // Fill neutral quantized input
    // =====================================================

    if (!fillFaceDetectorTestInput(
            *inputBuffers[0]
        )) {

        return 1;
    }

    // Explicitly clear outputs so we know graphExecute
    // will be responsible for writing them.
    clearOutputBuffers(
        outputBuffers
    );

    // =====================================================
    // Build contiguous tensor arrays
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
    // Print raw output samples
    //
    // No dequantization / decoding in 5G.1.
    // =====================================================

    std::cout
        << "[INFO] raw SCRFD outputs:\n";

    for (std::size_t i = 0;
         i < outputBuffers.size();
         ++i) {

        std::cout
            << "[INFO] output["
            << i
            << "] "
            << outputBuffers[i]->name()
            << '\n';

        printRawOutputSample(
            *outputBuffers[i]
        );
    }

    std::cout
        << "[PASS] 5G.1 SCRFD execution test complete\n";

    return 0;
}