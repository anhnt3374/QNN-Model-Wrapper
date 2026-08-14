#include "inference/qnn_backend.hpp"
#include "inference/qnn_quantization.hpp"
#include "models/face_detection_model.hpp"

#include <QnnTypes.h>

#include <opencv2/core.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace {

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
    if (actual != expected) {

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

bool expectFloatNear(
    const char* name,
    float actual,
    float expected,
    float tolerance
)
{
    if (std::fabs(
            actual - expected
        ) > tolerance) {

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
    // FaceDetectionModel
    // =====================================================

    models::FaceDetectionModel model(
        backend
    );

    if (!model.initialize(
            modelPath
        )) {

        std::cerr
            << "[ERROR] FaceDetectionModel initialize: "
            << model.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] FaceDetectionModel initialized\n";

    // =====================================================
    // Synthetic BGR image
    //
    // width  = 200
    // height = 100
    //
    // B = 10
    // G = 20
    // R = 30
    //
    // Expected resize:
    //
    // 200x100
    //     ↓
    // 640x320
    //
    // canvas:
    //
    // 640x640
    //
    // bottom 320 rows are black padding.
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
        << "[PASS] SCRFD preprocess succeeded\n";

    // =====================================================
    // Resize metadata
    // =====================================================

    const auto& info =
        model.preprocessInfo();

    if (!expectInt(
            "original width",
            info.originalWidth,
            200
        )) {

        return 1;
    }

    if (!expectInt(
            "original height",
            info.originalHeight,
            100
        )) {

        return 1;
    }

    if (!expectInt(
            "resized width",
            info.resizedWidth,
            640
        )) {

        return 1;
    }

    if (!expectInt(
            "resized height",
            info.resizedHeight,
            320
        )) {

        return 1;
    }

    if (!expectFloatNear(
            "det scale",
            info.detScale,
            3.2F,
            1e-6F
        )) {

        return 1;
    }

    // =====================================================
    // Inspect quantized QNN input
    // =====================================================

    const inference::QnnTensorBuffer*
        input =
            model.inputBuffer();

    if (input == nullptr) {

        std::cerr
            << "[ERROR] input buffer is null\n";

        return 1;
    }

    const auto* params =
        quantization(
            input->tensor()
        );

    if (params == nullptr) {

        std::cerr
            << "[ERROR] input quantization is null\n";

        return 1;
    }

    const float scale =
        params
            ->scaleOffsetEncoding
            .scale;

    const int32_t offset =
        params
            ->scaleOffsetEncoding
            .offset;

    const auto* data =
        static_cast<const uint16_t*>(
            input->data()
        );

    if (data == nullptr) {

        std::cerr
            << "[ERROR] input data pointer is null\n";

        return 1;
    }

    const float tolerance =
        scale * 1.1F;

    // =====================================================
    // First source pixel:
    //
    // BGR = [10,20,30]
    //
    // after BGR->RGB:
    //
    // RGB = [30,20,10]
    // =====================================================

    const float expectedR =
        (30.0F - 127.5F)
        /
        128.0F;

    const float expectedG =
        (20.0F - 127.5F)
        /
        128.0F;

    const float expectedB =
        (10.0F - 127.5F)
        /
        128.0F;

    const float actualR =
        inference::dequantizeScaleOffset(
            data[0],
            scale,
            offset
        );

    const float actualG =
        inference::dequantizeScaleOffset(
            data[1],
            scale,
            offset
        );

    const float actualB =
        inference::dequantizeScaleOffset(
            data[2],
            scale,
            offset
        );

    if (!expectFloatNear(
            "RGB channel R",
            actualR,
            expectedR,
            tolerance
        )) {

        return 1;
    }

    if (!expectFloatNear(
            "RGB channel G",
            actualG,
            expectedG,
            tolerance
        )) {

        return 1;
    }

    if (!expectFloatNear(
            "RGB channel B",
            actualB,
            expectedB,
            tolerance
        )) {

        return 1;
    }

    // =====================================================
    // Check padded area.
    //
    // y=400 is below resizedHeight=320,
    // therefore this pixel came from zero canvas.
    // =====================================================

    constexpr int paddedY = 400;
    constexpr int paddedX = 10;

    const uint64_t paddedIndex =
        (
            static_cast<uint64_t>(
                paddedY
            )
            *
            640ULL
            +
            static_cast<uint64_t>(
                paddedX
            )
        )
        *
        3ULL;

    const float expectedPadding =
        (0.0F - 127.5F)
        /
        128.0F;

    for (int channel = 0;
         channel < 3;
         ++channel) {

        const float actual =
            inference::dequantizeScaleOffset(
                data[
                    paddedIndex
                    +
                    static_cast<uint64_t>(
                        channel
                    )
                ],
                scale,
                offset
            );

        if (!expectFloatNear(
                "top-left letterbox padding",
                actual,
                expectedPadding,
                tolerance
            )) {

            return 1;
        }
    }

    std::cout
        << "[INFO] preprocess result:\n"
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
        << "       det_scale: "
        << info.detScale
        << '\n'
        << "       input elements: "
        << input->elementCount()
        << '\n';

    std::cout
        << "[PASS] 5H.1 FaceDetectionModel "
        << "preprocess test complete\n";

    return 0;
}