#include "inference/qnn_backend.hpp"
#include "inference/qnn_quantization.hpp"
#include "models/person_detection_model.hpp"

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
    const Qnn_DataType_t type =
        dataType(
            buffer.tensor()
        );

    if (type ==
        QNN_DATATYPE_FLOAT_32) {

        const auto* data =
            static_cast<const float*>(
                buffer.data()
            );

        if (data == nullptr) {
            return false;
        }

        value =
            data[index];

        return true;
    }

    if (type ==
        QNN_DATATYPE_UFIXED_POINT_16) {

        const auto* data =
            static_cast<const uint16_t*>(
                buffer.data()
            );

        const auto* params =
            quantization(
                buffer.tensor()
            );

        if (data == nullptr ||
            params == nullptr) {

            return false;
        }

        value =
            inference::dequantizeScaleOffset(
                data[index],
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

    if (backendPath == nullptr ||
        modelPath == nullptr) {

        std::cerr
            << "[ERROR] QNN environment missing\n";

        return 1;
    }

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
    // Synthetic image:
    //
    // 200x100
    //
    // B=10
    // G=20
    // R=30
    //
    // scale = 640 / 200 = 3.2
    //
    // resized = 640x320
    //
    // centered vertically:
    //
    // top    = 160
    // bottom = 160
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
        << "[PASS] person preprocess succeeded\n";

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
            640
        ) ||
        !expectInt(
            "resized height",
            info.resizedHeight,
            320
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
            160
        ) ||
        !expectInt(
            "pad bottom",
            info.padBottom,
            160
        )) {

        return 1;
    }

    if (!expectNear(
            "scale",
            info.scale,
            3.2F,
            1e-6F
        )) {

        return 1;
    }

    const auto* input =
        model.inputBuffer();

    if (input == nullptr) {

        std::cerr
            << "[ERROR] input buffer is null\n";

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

    // =====================================================
    // Padding pixel:
    //
    // x=0, y=0
    //
    // RGB = [114,114,114]
    //
    // normalized = 114/255
    // =====================================================

    const float expectedPadding =
        114.0F
        /
        255.0F;

    for (int channel = 0;
         channel < 3;
         ++channel) {

        float actual = 0.0F;

        if (!readRealValue(
                *input,
                static_cast<uint64_t>(
                    channel
                ),
                actual
            )) {

            std::cerr
                << "[ERROR] Cannot read padding input value\n";

            return 1;
        }

        float tolerance =
            1e-6F;

        const auto* params =
            quantization(
                input->tensor()
            );

        if (dataType(
                input->tensor()
            ) ==
                QNN_DATATYPE_UFIXED_POINT_16 &&
            params != nullptr) {

            tolerance =
                params
                    ->scaleOffsetEncoding
                    .scale
                *
                1.1F;
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
    // First content pixel:
    //
    // content starts at:
    //
    // x = 0
    // y = 160
    //
    // BGR = [10,20,30]
    // RGB = [30,20,10]
    //
    // normalized:
    //
    // R = 30/255
    // G = 20/255
    // B = 10/255
    // =====================================================

    constexpr uint64_t contentY =
        160;

    constexpr uint64_t contentX =
        0;

    const uint64_t baseIndex =
        (
            contentY
            *
            640ULL
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

    float tolerance =
        1e-6F;

    const auto* params =
        quantization(
            input->tensor()
        );

    if (dataType(
            input->tensor()
        ) ==
            QNN_DATATYPE_UFIXED_POINT_16 &&
        params != nullptr) {

        tolerance =
            params
                ->scaleOffsetEncoding
                .scale
            *
            1.1F;
    }

    for (uint64_t channel = 0;
         channel < 3;
         ++channel) {

        float actual = 0.0F;

        if (!readRealValue(
                *input,
                baseIndex + channel,
                actual
            )) {

            std::cerr
                << "[ERROR] Cannot read RGB input value\n";

            return 1;
        }

        if (!expectNear(
                "RGB normalized value",
                actual,
                expectedRgb[channel],
                tolerance
            )) {

            return 1;
        }
    }

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
        << "[PASS] 6P.2 PersonDetectionModel "
        << "preprocess test complete\n";

    return 0;
}