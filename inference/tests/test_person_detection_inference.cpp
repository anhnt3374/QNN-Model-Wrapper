#include "inference/qnn_backend.hpp"
#include "inference/qnn_quantization.hpp"
#include "models/person_detection_model.hpp"

#include <QnnTypes.h>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

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

bool readRealValue(
    const inference::QnnTensorBuffer& buffer,
    uint64_t index,
    float& value
)
{
    if (index >=
        buffer.elementCount()) {

        return false;
    }

    const Qnn_DataType_t type =
        dataType(
            buffer.tensor()
        );

    if (type ==
        QNN_DATATYPE_FLOAT_32) {

        const auto* raw =
            static_cast<const float*>(
                buffer.data()
            );

        if (raw == nullptr) {
            return false;
        }

        value =
            raw[index];

        return true;
    }

    if (type ==
        QNN_DATATYPE_UFIXED_POINT_16) {

        const auto* raw =
            static_cast<const uint16_t*>(
                buffer.data()
            );

        const auto* params =
            quantization(
                buffer.tensor()
            );

        if (raw == nullptr ||
            params == nullptr) {

            return false;
        }

        if (params->quantizationEncoding !=
            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {

            return false;
        }

        value =
            inference::dequantizeScaleOffset(
                raw[index],
                params
                    ->scaleOffsetEncoding
                    .scale,
                params
                    ->scaleOffsetEncoding
                    .offset
            );

        return true;
    }

    return false;
}

void printOutput(
    const inference::QnnTensorBuffer& buffer
)
{
    const Qnn_DataType_t type =
        dataType(
            buffer.tensor()
        );

    std::cout
        << "[INFO] output "
        << buffer.name()
        << '\n';

    std::cout
        << "       datatype: "
        << static_cast<uint32_t>(
            type
        )
        << '\n';

    std::cout
        << "       elements: "
        << buffer.elementCount()
        << '\n';

    std::cout
        << "       bytes: "
        << buffer.byteSize()
        << '\n';

    if (type ==
        QNN_DATATYPE_UFIXED_POINT_16) {

        const auto* params =
            quantization(
                buffer.tensor()
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

    const uint64_t sampleCount =
        std::min<uint64_t>(
            8,
            buffer.elementCount()
        );

    std::cout
        << "       values: [";

    for (uint64_t i = 0;
         i < sampleCount;
         ++i) {

        if (i > 0) {
            std::cout << ", ";
        }

        float value = 0.0F;

        if (!readRealValue(
                buffer,
                i,
                value
            )) {

            std::cout
                << "<read-error>";

            continue;
        }

        std::cout
            << std::fixed
            << std::setprecision(6)
            << value;
    }

    std::cout
        << "]\n"
        << std::defaultfloat;
}

bool validateOutputFinite(
    const inference::QnnTensorBuffer& buffer,
    uint64_t& finiteCount,
    uint64_t& nonZeroCount
)
{
    finiteCount = 0;
    nonZeroCount = 0;

    for (uint64_t i = 0;
         i < buffer.elementCount();
         ++i) {

        float value = 0.0F;

        if (!readRealValue(
                buffer,
                i,
                value
            )) {

            return false;
        }

        if (std::isfinite(
                value
            )) {

            ++finiteCount;
        }

        if (std::fabs(
                value
            ) >
            1e-12F) {

            ++nonZeroCount;
        }
    }

    return true;
}

} // namespace

int main(
    int argc,
    char** argv
)
{
    if (argc != 2) {

        std::cerr
            << "Usage:\n"
            << "  "
            << argv[0]
            << " <image.jpg>\n";

        return 1;
    }

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

    const char* imagePath =
        argv[1];

    // =====================================================
    // Load real image
    // =====================================================

    cv::Mat image =
        cv::imread(
            imagePath,
            cv::IMREAD_COLOR
        );

    if (image.empty()) {

        std::cerr
            << "[ERROR] Cannot read image: "
            << imagePath
            << '\n';

        return 1;
    }

    std::cout
        << "[INFO] image: "
        << imagePath
        << '\n';

    std::cout
        << "[INFO] image size: "
        << image.cols
        << "x"
        << image.rows
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
    // Person model
    // =====================================================

    models::PersonDetectionModel model(
        backend
    );

    if (!model.initialize(
            modelPath
        )) {

        std::cerr
            << "[ERROR] initialize: "
            << model.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] PersonDetectionModel initialized\n";

    // =====================================================
    // Preprocess real image
    // =====================================================

    if (!model.preprocess(
            image
        )) {

        std::cerr
            << "[ERROR] preprocess: "
            << model.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] real image preprocessed\n";

    const auto& info =
        model.preprocessInfo();

    std::cout
        << "[INFO] preprocess:\n";

    std::cout
        << "       original: "
        << info.originalWidth
        << "x"
        << info.originalHeight
        << '\n';

    std::cout
        << "       resized: "
        << info.resizedWidth
        << "x"
        << info.resizedHeight
        << '\n';

    std::cout
        << "       scale: "
        << info.scale
        << '\n';

    std::cout
        << "       padding: L="
        << info.padLeft
        << " R="
        << info.padRight
        << " T="
        << info.padTop
        << " B="
        << info.padBottom
        << '\n';

    // =====================================================
    // Execute on HTP
    // =====================================================

    std::cout
        << "[INFO] executing PersonDetectionModel...\n";

    if (!model.infer()) {

        std::cerr
            << "[ERROR] infer: "
            << model.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] PersonDetectionModel inference succeeded\n";

    // =====================================================
    // Outputs
    // =====================================================

    const auto* boxes =
        model.outputBufferByName(
            "boxes_out"
        );

    const auto* conf =
        model.outputBufferByName(
            "conf_out"
        );

    if (boxes == nullptr ||
        conf == nullptr) {

        std::cerr
            << "[ERROR] Required outputs are missing\n";

        return 1;
    }

    printOutput(
        *boxes
    );

    printOutput(
        *conf
    );

    // =====================================================
    // Validate full output buffers
    // =====================================================

    uint64_t boxesFinite = 0;
    uint64_t boxesNonZero = 0;

    if (!validateOutputFinite(
            *boxes,
            boxesFinite,
            boxesNonZero
        )) {

        std::cerr
            << "[ERROR] Cannot validate boxes_out\n";

        return 1;
    }

    uint64_t confFinite = 0;
    uint64_t confNonZero = 0;

    if (!validateOutputFinite(
            *conf,
            confFinite,
            confNonZero
        )) {

        std::cerr
            << "[ERROR] Cannot validate conf_out\n";

        return 1;
    }

    if (boxesFinite !=
        boxes->elementCount()) {

        std::cerr
            << "[ERROR] boxes_out contains non-finite values\n";

        return 1;
    }

    if (confFinite !=
        conf->elementCount()) {

        std::cerr
            << "[ERROR] conf_out contains non-finite values\n";

        return 1;
    }

    std::cout
        << "[PASS] all boxes_out values are finite\n";

    std::cout
        << "[PASS] all conf_out values are finite\n";

    std::cout
        << "[INFO] output statistics:\n";

    std::cout
        << "       boxes non-zero: "
        << boxesNonZero
        << " / "
        << boxes->elementCount()
        << '\n';

    std::cout
        << "       conf non-zero: "
        << confNonZero
        << " / "
        << conf->elementCount()
        << '\n';

    // boxes should not reasonably be completely empty after
    // a valid YOLO forward pass.
    if (boxesNonZero == 0) {

        std::cerr
            << "[ERROR] boxes_out is completely zero\n";

        return 1;
    }

    std::cout
        << "[PASS] person detector produced output data\n";

    std::cout
        << "[PASS] 6P.3 PersonDetectionModel "
        << "real-image inference test complete\n";

    return 0;
}