#include "inference/qnn_backend.hpp"
#include "inference/qnn_model.hpp"
#include "inference/qnn_tensor_buffer.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

namespace {

using TensorBufferPtr =
    std::unique_ptr<
        inference::QnnTensorBuffer
    >;

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
            << "       elements: "
            << buffer->elementCount()
            << '\n';

        std::cout
            << "       bytes / element: "
            << buffer->bytesPerElement()
            << '\n';

        std::cout
            << "       buffer bytes: "
            << buffer->byteSize()
            << '\n';

        std::cout
            << "       data pointer: "
            << buffer->data()
            << '\n';

        if (buffer->data() == nullptr) {

            std::cerr
                << "[ERROR] "
                << prefix
                << "["
                << i
                << "] has null buffer\n";

            return false;
        }

        buffers.push_back(
            std::move(buffer)
        );
    }

    return true;
}

uint64_t totalBufferBytes(
    const std::vector<TensorBufferPtr>& buffers
)
{
    uint64_t total = 0;

    for (const auto& buffer :
         buffers) {

        total +=
            buffer->byteSize();
    }

    return total;
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
            << "[ERROR] "
            << "QNN_BACKEND_PATH is not set\n";

        return 1;
    }

    if (modelPath == nullptr) {

        std::cerr
            << "[ERROR] "
            << "QNN_MODEL_PATH is not set\n";

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
    // QNN model
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
    // Finalize graph
    // =====================================================

    if (!model.finalizeGraphs()) {

        std::cerr
            << "[ERROR] finalizeGraphs: "
            << model.lastError()
            << '\n';

        return 1;
    }

    if (!model.graphsFinalized()) {

        std::cerr
            << "[ERROR] graphs are not finalized\n";

        return 1;
    }

    std::cout
        << "[PASS] QNN graphs finalized\n";

    for (uint32_t i = 0;
         i < model.graphCount();
         ++i) {

        std::cout
            << "[INFO] graph["
            << i
            << "] finalized: "
            << (
                model.graphFinalized(i)
                    ? "yes"
                    : "no"
            )
            << '\n';
    }

    // =====================================================
    // Graph metadata
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
        << "[INFO] input count: "
        << graph->numInputTensors
        << '\n';

    std::cout
        << "[INFO] output count: "
        << graph->numOutputTensors
        << '\n';

    // =====================================================
    // Input runtime buffers
    // =====================================================

    std::vector<TensorBufferPtr>
        inputBuffers;

    if (!createTensorBuffers(
            graph->inputTensors,
            graph->numInputTensors,
            inputBuffers,
            "input"
        )) {

        return 1;
    }

    // =====================================================
    // Output runtime buffers
    // =====================================================

    std::vector<TensorBufferPtr>
        outputBuffers;

    if (!createTensorBuffers(
            graph->outputTensors,
            graph->numOutputTensors,
            outputBuffers,
            "output"
        )) {

        return 1;
    }

    // =====================================================
    // Summary
    // =====================================================

    const uint64_t inputBytes =
        totalBufferBytes(
            inputBuffers
        );

    const uint64_t outputBytes =
        totalBufferBytes(
            outputBuffers
        );

    std::cout
        << "\n"
        << "[INFO] ==============================\n";

    std::cout
        << "[INFO] input buffer bytes: "
        << inputBytes
        << '\n';

    std::cout
        << "[INFO] output buffer bytes: "
        << outputBytes
        << '\n';

    std::cout
        << "[INFO] total runtime I/O bytes: "
        << (
            inputBytes +
            outputBytes
        )
        << '\n';

    std::cout
        << "[PASS] QNN tensor buffers ready\n";

    std::cout
        << "[PASS] QNN model ready for execution\n";

    return 0;
}