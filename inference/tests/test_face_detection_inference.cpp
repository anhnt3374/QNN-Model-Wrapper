#include "inference/qnn_backend.hpp"
#include "inference/qnn_quantization.hpp"
#include "models/face_detection_model.hpp"

#include <QnnTypes.h>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

namespace {

Qnn_DataType_t dataType(
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

bool printOutputSample(
    const inference::QnnTensorBuffer& buffer,
    std::size_t maxValues,
    uint64_t& totalNonZero
)
{
    if (dataType(
            buffer.tensor()
        ) !=
        QNN_DATATYPE_UFIXED_POINT_16) {

        std::cerr
            << "[ERROR] "
            << buffer.name()
            << " is not UFIXED_POINT_16\n";

        return false;
    }

    const auto* params =
        quantization(
            buffer.tensor()
        );

    if (params == nullptr) {

        std::cerr
            << "[ERROR] "
            << buffer.name()
            << " quantization is null\n";

        return false;
    }

    if (params->quantizationEncoding !=
        QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {

        std::cerr
            << "[ERROR] "
            << buffer.name()
            << " does not use SCALE_OFFSET\n";

        return false;
    }

    const auto* data =
        static_cast<const uint16_t*>(
            buffer.data()
        );

    if (data == nullptr) {

        std::cerr
            << "[ERROR] "
            << buffer.name()
            << " data pointer is null\n";

        return false;
    }

    const float scale =
        params
            ->scaleOffsetEncoding
            .scale;

    const int32_t offset =
        params
            ->scaleOffsetEncoding
            .offset;

    const std::size_t count =
        std::min<std::size_t>(
            maxValues,
            static_cast<std::size_t>(
                buffer.elementCount()
            )
        );

    uint64_t nonZero = 0;

    for (uint64_t i = 0;
         i < buffer.elementCount();
         ++i) {

        if (data[i] != 0) {
            ++nonZero;
        }
    }

    totalNonZero += nonZero;

    std::cout
        << "       elements: "
        << buffer.elementCount()
        << '\n';

    std::cout
        << "       scale: "
        << scale
        << '\n';

    std::cout
        << "       offset: "
        << offset
        << '\n';

    std::cout
        << "       non-zero raw values: "
        << nonZero
        << '\n';

    // =====================================================
    // Raw sample
    // =====================================================

    std::cout
        << "       raw: [";

    for (std::size_t i = 0;
         i < count;
         ++i) {

        if (i > 0) {
            std::cout << ", ";
        }

        std::cout
            << data[i];
    }

    std::cout
        << "]\n";

    // =====================================================
    // Dequantized sample
    // =====================================================

    std::cout
        << "       float: ["
        << std::fixed
        << std::setprecision(6);

    for (std::size_t i = 0;
         i < count;
         ++i) {

        if (i > 0) {
            std::cout << ", ";
        }

        std::cout
            << inference::dequantizeScaleOffset(
                   data[i],
                   scale,
                   offset
               );
    }

    std::cout
        << "]\n"
        << std::defaultfloat;

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
    // Load real image
    // =====================================================

    const char* imagePath =
        argv[1];

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
    // SCRFD
    // =====================================================

    models::FaceDetectionModel model(
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
        << "[PASS] FaceDetectionModel initialized\n";

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

    const auto& info =
        model.preprocessInfo();

    std::cout
        << "[PASS] real image preprocessed\n";

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
        << "       det_scale: "
        << info.detScale
        << '\n';

    // =====================================================
    // Execute model
    // =====================================================

    std::cout
        << "[INFO] executing FaceDetectionModel...\n";

    if (!model.infer()) {

        std::cerr
            << "[ERROR] infer: "
            << model.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] FaceDetectionModel inference succeeded\n";

    // =====================================================
    // Inspect all nine SCRFD output buffers
    // =====================================================

    if (model.outputCount() != 9) {

        std::cerr
            << "[ERROR] expected 9 outputs, got "
            << model.outputCount()
            << '\n';

        return 1;
    }

    uint64_t totalNonZero = 0;

    for (std::size_t i = 0;
         i < model.outputCount();
         ++i) {

        const auto* output =
            model.outputBuffer(
                i
            );

        if (output == nullptr) {

            std::cerr
                << "[ERROR] output["
                << i
                << "] is null\n";

            return 1;
        }

        std::cout
            << "[INFO] output["
            << i
            << "] "
            << output->name()
            << '\n';

        if (!printOutputSample(
                *output,
                8,
                totalNonZero
            )) {

            return 1;
        }
    }

    if (totalNonZero == 0) {

        std::cerr
            << "[ERROR] all SCRFD output buffers "
            << "are still zero\n";

        return 1;
    }

    std::cout
        << "[INFO] total non-zero output values: "
        << totalNonZero
        << '\n';

    std::cout
        << "[PASS] 5H.2 real-image SCRFD "
        << "inference test complete\n";

    return 0;
}