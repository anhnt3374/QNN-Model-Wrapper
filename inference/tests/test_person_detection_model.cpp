#include "inference/qnn_backend.hpp"
#include "models/person_detection_model.hpp"

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

void printTensor(
    const char* prefix,
    const inference::QnnTensorBuffer& buffer
)
{
    const Qnn_Tensor_t&
        tensor =
            buffer.tensor();

    std::cout
        << "[INFO] "
        << prefix
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
            "QNN_PERSON_MODEL_PATH"
        );

    if (backendPath == nullptr) {

        std::cerr
            << "[ERROR] QNN_BACKEND_PATH is not set\n";

        return 1;
    }

    if (modelPath == nullptr) {

        std::cerr
            << "[ERROR] QNN_PERSON_MODEL_PATH is not set\n";

        return 1;
    }

    std::cout
        << "[INFO] backend: "
        << backendPath
        << '\n';

    std::cout
        << "[INFO] person model: "
        << modelPath
        << '\n';

    // =====================================================
    // Shared QNN backend
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
    // PersonDetectionModel
    // =====================================================

    models::PersonDetectionModel model(
        backend
    );

    if (!model.initialize(
            modelPath
        )) {

        std::cerr
            << "[ERROR] PersonDetectionModel initialize: "
            << model.lastError()
            << '\n';

        return 1;
    }

    if (!model.ready()) {

        std::cerr
            << "[ERROR] PersonDetectionModel is not ready\n";

        return 1;
    }

    std::cout
        << "[PASS] PersonDetectionModel initialized\n";

    // =====================================================
    // Input
    // =====================================================

    const auto*
        input =
            model.inputBuffer();

    if (input == nullptr) {

        std::cerr
            << "[ERROR] Person input buffer is null\n";

        return 1;
    }

    printTensor(
        "input",
        *input
    );

    // =====================================================
    // Outputs
    // =====================================================

    if (model.outputCount() != 2) {

        std::cerr
            << "[ERROR] Expected 2 person outputs, got "
            << model.outputCount()
            << '\n';

        return 1;
    }

    const auto*
        boxes =
            model.outputBufferByName(
                "boxes_out"
            );

    const auto*
        conf =
            model.outputBufferByName(
                "conf_out"
            );

    if (boxes == nullptr ||
        conf == nullptr) {

        std::cerr
            << "[ERROR] Required person outputs are missing\n";

        return 1;
    }

    printTensor(
        "boxes_out",
        *boxes
    );

    printTensor(
        "conf_out",
        *conf
    );

    std::cout
        << "[PASS] Person detector tensor contract validated\n";

    std::cout
        << "[PASS] Person detector runtime buffers ready\n";

    std::cout
        << "[PASS] 6P.1 PersonDetectionModel "
        << "contract test complete\n";

    return 0;
}