#include "inference/qnn_backend.hpp"
#include "models/fire_smoke_detection_model.hpp"

#include <QnnTypes.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>

namespace {

Qnn_DataType_t dataType(
    const Qnn_Tensor_t& tensor
)
{
    switch (tensor.version) {

    case QNN_TENSOR_VERSION_1:
        return
            tensor.v1.dataType;

    case QNN_TENSOR_VERSION_2:
        return
            tensor.v2.dataType;

    default:
        return
            QNN_DATATYPE_UNDEFINED;
    }
}

uint32_t rank(
    const Qnn_Tensor_t& tensor
)
{
    switch (tensor.version) {

    case QNN_TENSOR_VERSION_1:
        return
            tensor.v1.rank;

    case QNN_TENSOR_VERSION_2:
        return
            tensor.v2.rank;

    default:
        return 0;
    }
}

const uint32_t*
dimensions(
    const Qnn_Tensor_t& tensor
)
{
    switch (tensor.version) {

    case QNN_TENSOR_VERSION_1:
        return
            tensor.v1.dimensions;

    case QNN_TENSOR_VERSION_2:
        return
            tensor.v2.dimensions;

    default:
        return nullptr;
    }
}

const Qnn_QuantizeParams_t*
quantization(
    const Qnn_Tensor_t& tensor
)
{
    switch (tensor.version) {

    case QNN_TENSOR_VERSION_1:
        return
            &tensor.v1.quantizeParams;

    case QNN_TENSOR_VERSION_2:
        return
            &tensor.v2.quantizeParams;

    default:
        return nullptr;
    }
}

void printTensor(
    const char* label,
    const inference::QnnTensorBuffer& buffer
)
{
    const Qnn_Tensor_t&
        tensor =
            buffer.tensor();

    std::cout
        << "[INFO] "
        << label
        << '\n';

    std::cout
        << "       name: "
        << buffer.name()
        << '\n';

    std::cout
        << "       datatype: "
        << static_cast<uint32_t>(
            dataType(
                tensor
            )
        )
        << '\n';

    std::cout
        << "       rank: "
        << rank(
            tensor
        )
        << '\n';

    const uint32_t*
        dims =
            dimensions(
                tensor
            );

    std::cout
        << "       shape: [";

    if (dims != nullptr) {

        for (uint32_t i = 0;
             i < rank(
                 tensor
             );
             ++i) {

            if (i > 0) {
                std::cout << ", ";
            }

            std::cout
                << dims[i];
        }
    }

    std::cout
        << "]\n";

    std::cout
        << "       elements: "
        << buffer.elementCount()
        << '\n';

    std::cout
        << "       bytes: "
        << buffer.byteSize()
        << '\n';

    if (dataType(
            tensor
        ) ==
        QNN_DATATYPE_UFIXED_POINT_16) {

        const auto* params =
            quantization(
                tensor
            );

        if (params != nullptr &&
            params->quantizationEncoding ==
                QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {

            std::cout
                << "       scale: "
                << params
                    ->scaleOffsetEncoding
                    .scale
                << '\n';

            std::cout
                << "       offset: "
                << params
                    ->scaleOffsetEncoding
                    .offset
                << '\n';
        }
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
            "QNN_FIRE_SMOKE_MODEL_PATH"
        );

    if (backendPath == nullptr) {

        std::cerr
            << "[ERROR] QNN_BACKEND_PATH is not set\n";

        return 1;
    }

    if (modelPath == nullptr) {

        std::cerr
            << "[ERROR] QNN_FIRE_SMOKE_MODEL_PATH is not set\n";

        return 1;
    }

    std::cout
        << "[INFO] backend: "
        << backendPath
        << '\n';

    std::cout
        << "[INFO] FireSmoke model: "
        << modelPath
        << '\n';

    // =====================================================
    // Shared backend
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

    if (!backend.loadProviders()) {

        std::cerr
            << "[ERROR] loadProviders: "
            << backend.lastError()
            << '\n';

        return 1;
    }

    if (!backend.selectInterface()) {

        std::cerr
            << "[ERROR] selectInterface: "
            << backend.lastError()
            << '\n';

        return 1;
    }

    if (!backend.createBackend()) {

        std::cerr
            << "[ERROR] createBackend: "
            << backend.lastError()
            << '\n';

        return 1;
    }

    if (!backend.createDevice()) {

        std::cerr
            << "[ERROR] createDevice: "
            << backend.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] QNN backend ready\n";

    // =====================================================
    // FireSmoke model
    // =====================================================

    models::FireSmokeDetectionModel model(
        backend
    );

    if (!model.initialize(
            modelPath
        )) {

        std::cerr
            << "[ERROR] FireSmoke initialize: "
            << model.lastError()
            << '\n';

        return 1;
    }

    if (!model.ready()) {

        std::cerr
            << "[ERROR] FireSmoke model is not ready\n";

        return 1;
    }

    std::cout
        << "[PASS] FireSmokeDetectionModel initialized\n";

    // =====================================================
    // Input
    // =====================================================

    const auto* input =
        model.inputBuffer();

    if (input == nullptr) {

        std::cerr
            << "[ERROR] FireSmoke input buffer is null\n";

        return 1;
    }

    printTensor(
        "input",
        *input
    );

    // =====================================================
    // Outputs
    // =====================================================

    if (model.outputCount() != 3) {

        std::cerr
            << "[ERROR] Expected 3 FireSmoke outputs, got "
            << model.outputCount()
            << '\n';

        return 1;
    }

    const auto* outputS4 =
        model.outputBufferByName(
            "output_s4"
        );

    const auto* outputS8 =
        model.outputBufferByName(
            "output_s8"
        );

    const auto* outputS16 =
        model.outputBufferByName(
            "output_s16"
        );

    if (outputS4 == nullptr ||
        outputS8 == nullptr ||
        outputS16 == nullptr) {

        std::cerr
            << "[ERROR] Required FireSmoke outputs are missing\n";

        return 1;
    }

    printTensor(
        "output_s4",
        *outputS4
    );

    printTensor(
        "output_s8",
        *outputS8
    );

    printTensor(
        "output_s16",
        *outputS16
    );

    std::cout
        << "[PASS] FireSmoke tensor contract validated\n";

    std::cout
        << "[PASS] FireSmoke persistent buffers ready\n";

    std::cout
        << "[PASS] 7F.1 FireSmokeDetectionModel "
        << "contract test complete\n";

    return 0;
}