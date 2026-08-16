#include "inference/qnn_backend.hpp"
#include "models/face_embedding_model.hpp"

#include <QnnTypes.h>

#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
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

        return
            nullptr;
    }
}

void printTensorInfo(
    const char* label,
    const inference::QnnTensorBuffer& buffer
)
{
    const Qnn_Tensor_t& tensor =
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

float calculateNorm(
    const models::FaceEmbeddingResult& result
)
{
    double sum =
        0.0;

    for (const float value :
         result.values) {

        sum +=
            static_cast<double>(
                value
            )
            *
            static_cast<double>(
                value
            );
    }

    return
        static_cast<float>(
            std::sqrt(
                sum
            )
        );
}

} // namespace

int main(
    int argc,
    char** argv
)
{
    if (argc != 3) {

        std::cerr
            << "Usage:\n"
            << "  "
            << argv[0]
            << " <aligned_face.jpg>"
            << " <embedding.txt>\n";

        return 1;
    }

    const char* backendPath =
        std::getenv(
            "QNN_BACKEND_PATH"
        );

    const char* modelPath =
        std::getenv(
            "QNN_FACE_EMBEDDING_MODEL_PATH"
        );

    if (backendPath == nullptr) {

        std::cerr
            << "[ERROR] QNN_BACKEND_PATH is not set\n";

        return 1;
    }

    if (modelPath == nullptr) {

        std::cerr
            << "[ERROR] QNN_FACE_EMBEDDING_MODEL_PATH "
            << "is not set\n";

        return 1;
    }

    const char* imagePath =
        argv[1];

    const char* outputPath =
        argv[2];

    // =====================================================
    // Read face image
    // =====================================================

    cv::Mat face =
        cv::imread(
            imagePath,
            cv::IMREAD_COLOR
        );

    if (face.empty()) {

        std::cerr
            << "[ERROR] Cannot read face image: "
            << imagePath
            << '\n';

        return 1;
    }

    std::cout
        << "[INFO] face image: "
        << imagePath
        << '\n';

    std::cout
        << "[INFO] original size: "
        << face.cols
        << "x"
        << face.rows
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
    // FaceEmbeddingModel
    // =====================================================

    models::FaceEmbeddingModel model(
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

    if (!model.ready()) {

        std::cerr
            << "[ERROR] FaceEmbeddingModel is not ready\n";

        return 1;
    }

    std::cout
        << "[PASS] FaceEmbeddingModel initialized\n";

    // =====================================================
    // Print tensor contract
    // =====================================================

    const auto* input =
        model.inputBuffer();

    const auto* output =
        model.outputBuffer();

    if (input == nullptr ||
        output == nullptr) {

        std::cerr
            << "[ERROR] FaceEmbedding runtime buffers "
            << "are missing\n";

        return 1;
    }

    printTensorInfo(
        "input",
        *input
    );

    printTensorInfo(
        "embedding",
        *output
    );

    // =====================================================
    // Complete end-to-end extraction
    // =====================================================

    models::FaceEmbeddingResult result;

    if (!model.extract(
            face,
            result
        )) {

        std::cerr
            << "[ERROR] extract: "
            << model.lastError()
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] FaceEmbeddingModel extract succeeded\n";

    // =====================================================
    // Preprocess information
    // =====================================================

    const auto& info =
        model.preprocessInfo();

    if (info.resizedWidth != 112 ||
        info.resizedHeight != 112) {

        std::cerr
            << "[ERROR] FaceEmbedding resize is not 112x112\n";

        return 1;
    }

    std::cout
        << "[PASS] face resized to 112x112\n";

    // =====================================================
    // Validate all 512 values
    // =====================================================

    if (result.values.size() !=
        models::FaceEmbeddingModel::EMBEDDING_DIM) {

        std::cerr
            << "[ERROR] Embedding dimension is not 512\n";

        return 1;
    }

    float minValue =
        std::numeric_limits<float>::infinity();

    float maxValue =
        -std::numeric_limits<float>::infinity();

    for (const float value :
         result.values) {

        if (!std::isfinite(
                value
            )) {

            std::cerr
                << "[ERROR] Embedding contains "
                << "non-finite value\n";

            return 1;
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

    std::cout
        << "[PASS] all 512 embedding values are finite\n";

    // =====================================================
    // Raw norm
    // =====================================================

    if (!std::isfinite(
            result.rawL2Norm
        ) ||
        result.rawL2Norm <= 1e-12F) {

        std::cerr
            << "[ERROR] Invalid raw embedding norm: "
            << result.rawL2Norm
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] raw embedding norm is valid\n";

    // =====================================================
    // Normalized embedding must have ||x|| ~= 1
    // =====================================================

    const float normalizedNorm =
        calculateNorm(
            result
        );

    if (!std::isfinite(
            normalizedNorm
        ) ||
        std::fabs(
            normalizedNorm - 1.0F
        ) >
        1e-4F) {

        std::cerr
            << "[ERROR] Normalized embedding norm: "
            << normalizedNorm
            << '\n';

        return 1;
    }

    std::cout
        << "[PASS] embedding L2 norm is approximately 1\n";

    // =====================================================
    // Print summary
    // =====================================================

    std::cout
        << std::fixed
        << std::setprecision(9);

    std::cout
        << "[INFO] embedding summary:\n";

    std::cout
        << "       dimension: "
        << result.values.size()
        << '\n';

    std::cout
        << "       raw L2 norm: "
        << result.rawL2Norm
        << '\n';

    std::cout
        << "       normalized norm: "
        << normalizedNorm
        << '\n';

    std::cout
        << "       min: "
        << minValue
        << '\n';

    std::cout
        << "       max: "
        << maxValue
        << '\n';

    std::cout
        << "[INFO] first 16 normalized values:\n"
        << "       ";

    for (std::size_t i = 0;
         i < 16;
         ++i) {

        if (i > 0) {

            std::cout
                << ", ";
        }

        std::cout
            << result.values[i];
    }

    std::cout
        << '\n'
        << std::defaultfloat;

    // =====================================================
    // Save text embedding
    //
    // One normalized value per line.
    // =====================================================

    std::ofstream file(
        outputPath
    );

    if (!file) {

        std::cerr
            << "[ERROR] Cannot open embedding output: "
            << outputPath
            << '\n';

        return 1;
    }

    file
        << std::fixed
        << std::setprecision(9);

    for (const float value :
         result.values) {

        file
            << value
            << '\n';
    }

    file.close();

    if (!file) {

        std::cerr
            << "[ERROR] Failed writing embedding file\n";

        return 1;
    }

    std::cout
        << "[PASS] normalized embedding saved\n";

    std::cout
        << "[INFO] embedding file: "
        << outputPath
        << '\n';

    std::cout
        << "[PASS] FaceEmbeddingModel "
        << "end-to-end test complete\n";

    return 0;
}