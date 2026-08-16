#include "inference/qnn_backend.hpp"
#include "inference/qnn_quantization.hpp"
#include "models/fire_smoke_detection_model.hpp"

#include <QnnTypes.h>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>

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
            10,
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

bool validateOutput(
    const inference::QnnTensorBuffer& buffer,
    uint64_t& finiteCount,
    uint64_t& nonZeroCount,
    float& minValue,
    float& maxValue
)
{
    finiteCount = 0;
    nonZeroCount = 0;

    minValue =
        std::numeric_limits<float>::infinity();

    maxValue =
        -std::numeric_limits<float>::infinity();

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

        if (!std::isfinite(
                value
            )) {

            continue;
        }

        ++finiteCount;

        if (std::fabs(
                value
            ) >
            1e-12F) {

            ++nonZeroCount;
        }

        minValue =
            std::min(
                minValue,
                value
            );

        maxValue =
            std::max(
                maxValue,
                value
            );
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
            << " <firesmoke.jpg>\n";

        return 1;
    }

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

    const char* imagePath =
        argv[1];

    // =====================================================
    // Load image
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
    // Backend
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
            << "[ERROR] initialize: "
            << model.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] FireSmokeDetectionModel initialized\n";

    // =====================================================
    // Preprocess
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
    // HTP graphExecute
    // =====================================================

    std::cout
        << "[INFO] executing FireSmokeDetectionModel...\n";

    if (!model.infer()) {

        std::cerr
            << "[ERROR] infer: "
            << model.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] FireSmokeDetectionModel inference succeeded\n";

    // =====================================================
    // Get output heads
    // =====================================================

    const auto* s4 =
        model.outputBufferByName(
            "output_s4"
        );

    const auto* s8 =
        model.outputBufferByName(
            "output_s8"
        );

    const auto* s16 =
        model.outputBufferByName(
            "output_s16"
        );

    if (s4 == nullptr ||
        s8 == nullptr ||
        s16 == nullptr) {

        std::cerr
            << "[ERROR] FireSmoke outputs are missing\n";

        return 1;
    }

    printOutput(
        *s4
    );

    printOutput(
        *s8
    );

    printOutput(
        *s16
    );

    // =====================================================
    // Validate outputs
    // =====================================================

    const inference::QnnTensorBuffer*
        outputs[3] = {
            s4,
            s8,
            s16
        };

    for (const auto* output :
         outputs) {

        uint64_t finiteCount = 0;
        uint64_t nonZeroCount = 0;

        float minValue = 0.0F;
        float maxValue = 0.0F;

        if (!validateOutput(
                *output,
                finiteCount,
                nonZeroCount,
                minValue,
                maxValue
            )) {

            std::cerr
                << "[ERROR] Cannot validate "
                << output->name()
                << '\n';

            return 1;
        }

        if (finiteCount !=
            output->elementCount()) {

            std::cerr
                << "[ERROR] "
                << output->name()
                << " contains non-finite values\n";

            return 1;
        }

        if (nonZeroCount == 0) {

            std::cerr
                << "[ERROR] "
                << output->name()
                << " is completely zero\n";

            return 1;
        }

        std::cout
            << "[PASS] "
            << output->name()
            << " values are finite\n";

        std::cout
            << "[INFO] "
            << output->name()
            << " statistics:\n";

        std::cout
            << "       non-zero: "
            << nonZeroCount
            << " / "
            << output->elementCount()
            << '\n';

        std::cout
            << "       min: "
            << minValue
            << '\n';

        std::cout
            << "       max: "
            << maxValue
            << '\n';
    }

    std::cout
        << "[PASS] all FireSmoke heads produced output data\n";

    std::cout
        << "[PASS] 7F.3 FireSmokeDetectionModel "
        << "real-image inference test complete\n";

    return 0;
}