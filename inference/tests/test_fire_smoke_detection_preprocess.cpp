#include "inference/qnn_backend.hpp"
#include "inference/qnn_quantization.hpp"
#include "models/fire_smoke_detection_model.hpp"

#include <QnnTypes.h>

#include <opencv2/core.hpp>

#include <cmath>
#include <cstdint>
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

bool expectInt(
    const char* name,
    int actual,
    int expected
)
{
    if (actual !=
        expected) {

        std::cerr
            << "[ERROR] "
            << name
            << ": expected="
            << expected
            << ", actual="
            << actual
            << '\n';

        return false;
    }

    std::cout
        << "[PASS] "
        << name
        << '\n';

    return true;
}

bool expectNear(
    const char* name,
    float actual,
    float expected,
    float tolerance
)
{
    if (std::fabs(
            actual - expected
        ) >
        tolerance) {

        std::cerr
            << "[ERROR] "
            << name
            << ": expected="
            << expected
            << ", actual="
            << actual
            << ", tolerance="
            << tolerance
            << '\n';

        return false;
    }

    std::cout
        << "[PASS] "
        << name
        << '\n';

    return true;
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

float inputTolerance(
    const inference::QnnTensorBuffer& input
)
{
    if (dataType(
            input.tensor()
        ) ==
        QNN_DATATYPE_FLOAT_32) {

        return
            1e-6F;
    }

    const auto* params =
        quantization(
            input.tensor()
        );

    if (params == nullptr ||
        params->quantizationEncoding !=
            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {

        return
            1e-5F;
    }

    return
        params
            ->scaleOffsetEncoding
            .scale
        *
        1.1F;
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

    // =====================================================
    // Backend
    // =====================================================

    inference::QnnBackend backend;

    if (!backend.loadLibrary(
            backendPath
        ) ||
        !backend.loadProviders() ||
        !backend.selectInterface() ||
        !backend.createBackend() ||
        !backend.createDevice()) {

        std::cerr
            << "[ERROR] Cannot initialize QNN backend: "
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
    // Synthetic image:
    //
    // width  = 200
    // height = 100
    //
    // BGR = [10,20,30]
    //
    // scale:
    //
    // 320 / 200 = 1.6
    //
    // resized:
    //
    // 320x160
    //
    // vertical centered padding:
    //
    // top    = 80
    // bottom = 80
    // =====================================================

    cv::Mat image(
        100,
        200,
        CV_8UC3,
        cv::Scalar(
            10,
            20,
            30
        )
    );

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
        << "[PASS] FireSmoke preprocess succeeded\n";

    const auto& info =
        model.preprocessInfo();

    if (!expectInt(
            "original width",
            info.originalWidth,
            200
        ) ||
        !expectInt(
            "original height",
            info.originalHeight,
            100
        ) ||
        !expectInt(
            "resized width",
            info.resizedWidth,
            320
        ) ||
        !expectInt(
            "resized height",
            info.resizedHeight,
            160
        ) ||
        !expectInt(
            "pad left",
            info.padLeft,
            0
        ) ||
        !expectInt(
            "pad right",
            info.padRight,
            0
        ) ||
        !expectInt(
            "pad top",
            info.padTop,
            80
        ) ||
        !expectInt(
            "pad bottom",
            info.padBottom,
            80
        )) {

        return 1;
    }

    if (!expectNear(
            "scale",
            info.scale,
            1.6F,
            1e-6F
        )) {

        return 1;
    }

    const auto* input =
        model.inputBuffer();

    if (input == nullptr) {

        std::cerr
            << "[ERROR] FireSmoke input buffer is null\n";

        return 1;
    }

    std::cout
        << "[INFO] input datatype: "
        << static_cast<uint32_t>(
               dataType(
                   input->tensor()
               )
           )
        << '\n';

    if (dataType(
            input->tensor()
        ) ==
        QNN_DATATYPE_UFIXED_POINT_16) {

        const auto* params =
            quantization(
                input->tensor()
            );

        if (params != nullptr &&
            params->quantizationEncoding ==
                QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {

            std::cout
                << "[INFO] input quantization:\n"
                << "       scale: "
                << params
                    ->scaleOffsetEncoding
                    .scale
                << '\n'
                << "       offset: "
                << params
                    ->scaleOffsetEncoding
                    .offset
                << '\n';
        }
    }

    const float tolerance =
        inputTolerance(
            *input
        );

    // =====================================================
    // Padding pixel:
    //
    // [114,114,114] / 255
    // =====================================================

    const float expectedPadding =
        114.0F
        /
        255.0F;

    for (uint64_t channel = 0;
         channel < 3;
         ++channel) {

        float actual =
            0.0F;

        if (!readRealValue(
                *input,
                channel,
                actual
            )) {

            std::cerr
                << "[ERROR] Cannot read padding value\n";

            return 1;
        }

        if (!expectNear(
                "padding normalized value",
                actual,
                expectedPadding,
                tolerance
            )) {

            return 1;
        }
    }

    // =====================================================
    // First actual image pixel:
    //
    // content begins:
    //
    // y = 80
    // x = 0
    //
    // input BGR:
    //
    // [10,20,30]
    //
    // after BGR->RGB:
    //
    // [30,20,10]
    //
    // normalized:
    //
    // [30/255,20/255,10/255]
    // =====================================================

    constexpr uint64_t contentY =
        80;

    constexpr uint64_t contentX =
        0;

    const uint64_t baseIndex =
        (
            contentY
            *
            320ULL
            +
            contentX
        )
        *
        3ULL;

    const float expectedRgb[3] = {
        30.0F / 255.0F,
        20.0F / 255.0F,
        10.0F / 255.0F
    };

    for (uint64_t channel = 0;
         channel < 3;
         ++channel) {

        float actual =
            0.0F;

        if (!readRealValue(
                *input,
                baseIndex + channel,
                actual
            )) {

            std::cerr
                << "[ERROR] Cannot read RGB value\n";

            return 1;
        }

        if (!expectNear(
                "RGB normalized value",
                actual,
                expectedRgb[
                    channel
                ],
                tolerance
            )) {

            return 1;
        }
    }

    // =====================================================
    // Summary
    // =====================================================

    std::cout
        << "[INFO] preprocess geometry:\n"
        << "       original: "
        << info.originalWidth
        << "x"
        << info.originalHeight
        << '\n'
        << "       resized: "
        << info.resizedWidth
        << "x"
        << info.resizedHeight
        << '\n'
        << "       scale: "
        << info.scale
        << '\n'
        << "       padding: L="
        << info.padLeft
        << " R="
        << info.padRight
        << " T="
        << info.padTop
        << " B="
        << info.padBottom
        << '\n';

    std::cout
        << "[PASS] 7F.2 FireSmokeDetectionModel "
        << "preprocess test complete\n";

    return 0;
}