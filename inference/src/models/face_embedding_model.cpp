#include "models/face_embedding_model.hpp"

#include "inference/qnn_quantization.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdint>
#include <sstream>

namespace models {

FaceEmbeddingModel::FaceEmbeddingModel(
    inference::QnnBackend& backend
) noexcept
    : model_(backend)
{
}

FaceEmbeddingModel::~FaceEmbeddingModel()
{
    shutdown();
}

// =========================================================
// STEP 1
//
// Load / compose / finalize / validate / allocate
// =========================================================

bool FaceEmbeddingModel::initialize(
    const std::string& modelPath
)
{
    shutdown();

    lastError_.clear();

    // =====================================================
    // Load generated QNN model .so
    // =====================================================

    if (!model_.load(
            modelPath
        )) {

        lastError_ =
            "Failed to load FaceEmbedding model: "
            + model_.lastError();

        return false;
    }

    // =====================================================
    // Compose graph
    // =====================================================

    if (!model_.composeGraphs()) {

        lastError_ =
            "Failed to compose FaceEmbedding graph: "
            + model_.lastError();

        shutdown();

        return false;
    }

    if (model_.graphCount() != 1) {

        std::ostringstream oss;

        oss
            << "FaceEmbedding expected exactly 1 graph, got "
            << model_.graphCount();

        lastError_ =
            oss.str();

        shutdown();

        return false;
    }

    // =====================================================
    // Finalize graph
    // =====================================================

    if (!model_.finalizeGraphs()) {

        lastError_ =
            "Failed to finalize FaceEmbedding graph: "
            + model_.lastError();

        shutdown();

        return false;
    }

    const auto* graph =
        model_.graphInfo(
            0
        );

    if (graph == nullptr) {

        lastError_ =
            "FaceEmbedding graph metadata is null";

        shutdown();

        return false;
    }

    if (graph->numInputTensors != 1) {

        std::ostringstream oss;

        oss
            << "FaceEmbedding expected 1 input tensor, got "
            << graph->numInputTensors;

        lastError_ =
            oss.str();

        shutdown();

        return false;
    }

    if (graph->numOutputTensors != 1) {

        std::ostringstream oss;

        oss
            << "FaceEmbedding expected 1 output tensor, got "
            << graph->numOutputTensors;

        lastError_ =
            oss.str();

        shutdown();

        return false;
    }

    // =====================================================
    // Persistent buffers
    // =====================================================

    if (!allocateRuntimeBuffers()) {

        shutdown();

        return false;
    }

    // =====================================================
    // Tensor contract
    // =====================================================

    if (!validateModelContract()) {

        shutdown();

        return false;
    }

    // =====================================================
    // Persistent graphExecute descriptors
    // =====================================================

    if (!buildExecutionTensorArrays()) {

        shutdown();

        return false;
    }

    lastError_.clear();

    return true;
}

bool FaceEmbeddingModel::allocateRuntimeBuffers()
{
    const auto* graph =
        model_.graphInfo(
            0
        );

    if (graph == nullptr) {

        lastError_ =
            "Cannot allocate FaceEmbedding buffers: "
            "graph is null";

        return false;
    }

    inputBuffers_.clear();

    outputBuffers_.clear();

    inputBuffers_.reserve(
        graph->numInputTensors
    );

    outputBuffers_.reserve(
        graph->numOutputTensors
    );

    // =====================================================
    // Inputs
    // =====================================================

    for (uint32_t i = 0;
         i < graph->numInputTensors;
         ++i) {

        auto buffer =
            std::make_unique<
                inference::QnnTensorBuffer
            >();

        if (!buffer->initialize(
                graph->inputTensors[i]
            )) {

            std::ostringstream oss;

            oss
                << "Failed to allocate FaceEmbedding input["
                << i
                << "]: "
                << buffer->lastError();

            lastError_ =
                oss.str();

            return false;
        }

        inputBuffers_.push_back(
            std::move(
                buffer
            )
        );
    }

    // =====================================================
    // Outputs
    // =====================================================

    for (uint32_t i = 0;
         i < graph->numOutputTensors;
         ++i) {

        auto buffer =
            std::make_unique<
                inference::QnnTensorBuffer
            >();

        if (!buffer->initialize(
                graph->outputTensors[i]
            )) {

            std::ostringstream oss;

            oss
                << "Failed to allocate FaceEmbedding output["
                << i
                << "]: "
                << buffer->lastError();

            lastError_ =
                oss.str();

            return false;
        }

        outputBuffers_.push_back(
            std::move(
                buffer
            )
        );
    }

    return true;
}

bool FaceEmbeddingModel::validateModelContract()
{
    if (inputBuffers_.size() != 1) {

        lastError_ =
            "FaceEmbedding runtime input count must be 1";

        return false;
    }

    if (outputBuffers_.size() != 1) {

        lastError_ =
            "FaceEmbedding runtime output count must be 1";

        return false;
    }

    // =====================================================
    // Input:
    //
    // name  = input
    // shape = [1,112,112,3]
    // =====================================================

    const auto& input =
        *inputBuffers_[0];

    if (input.name() !=
        "input") {

        std::ostringstream oss;

        oss
            << "Expected FaceEmbedding input tensor 'input', got '"
            << input.name()
            << "'";

        lastError_ =
            oss.str();

        return false;
    }

    const Qnn_Tensor_t& inputTensor =
        input.tensor();

    if (rank(
            inputTensor
        ) != 4) {

        lastError_ =
            "FaceEmbedding input rank must be 4";

        return false;
    }

    const uint32_t* inputDims =
        dimensions(
            inputTensor
        );

    if (inputDims == nullptr) {

        lastError_ =
            "FaceEmbedding input dimensions are null";

        return false;
    }

    if (inputDims[0] != 1 ||
        inputDims[1] != INPUT_HEIGHT ||
        inputDims[2] != INPUT_WIDTH ||
        inputDims[3] != INPUT_CHANNELS) {

        std::ostringstream oss;

        oss
            << "Unexpected FaceEmbedding input shape: ["
            << inputDims[0]
            << ", "
            << inputDims[1]
            << ", "
            << inputDims[2]
            << ", "
            << inputDims[3]
            << "]";

        lastError_ =
            oss.str();

        return false;
    }

    const Qnn_DataType_t inputType =
        dataType(
            inputTensor
        );

    // qnn-net-run default may expose float input,
    // while direct graph/native execution may expose
    // quantized UFIXED_POINT_16.
    if (inputType !=
            QNN_DATATYPE_FLOAT_32 &&
        inputType !=
            QNN_DATATYPE_UFIXED_POINT_16) {

        std::ostringstream oss;

        oss
            << "Unsupported FaceEmbedding input datatype: "
            << static_cast<uint32_t>(
                   inputType
               );

        lastError_ =
            oss.str();

        return false;
    }

    // =====================================================
    // Output:
    //
    // name  = embedding
    // shape = [1,512]
    // =====================================================

    const auto& output =
        *outputBuffers_[0];

    if (output.name() !=
        "embedding") {

        std::ostringstream oss;

        oss
            << "Expected FaceEmbedding output tensor "
            << "'embedding', got '"
            << output.name()
            << "'";

        lastError_ =
            oss.str();

        return false;
    }

    const Qnn_Tensor_t& outputTensor =
        output.tensor();

    if (rank(
            outputTensor
        ) != 2) {

        lastError_ =
            "FaceEmbedding output rank must be 2";

        return false;
    }

    const uint32_t* outputDims =
        dimensions(
            outputTensor
        );

    if (outputDims == nullptr) {

        lastError_ =
            "FaceEmbedding output dimensions are null";

        return false;
    }

    if (outputDims[0] != 1 ||
        outputDims[1] != EMBEDDING_DIM) {

        std::ostringstream oss;

        oss
            << "Unexpected FaceEmbedding output shape: ["
            << outputDims[0]
            << ", "
            << outputDims[1]
            << "]";

        lastError_ =
            oss.str();

        return false;
    }

    if (output.elementCount() !=
        EMBEDDING_DIM) {

        std::ostringstream oss;

        oss
            << "Unexpected embedding element count: "
            << output.elementCount();

        lastError_ =
            oss.str();

        return false;
    }

    const Qnn_DataType_t outputType =
        dataType(
            outputTensor
        );

    if (outputType !=
            QNN_DATATYPE_FLOAT_32 &&
        outputType !=
            QNN_DATATYPE_UFIXED_POINT_16) {

        std::ostringstream oss;

        oss
            << "Unsupported FaceEmbedding output datatype: "
            << static_cast<uint32_t>(
                   outputType
               );

        lastError_ =
            oss.str();

        return false;
    }

    lastError_.clear();

    return true;
}

bool FaceEmbeddingModel::buildExecutionTensorArrays()
{
    executionInputTensors_.clear();

    executionOutputTensors_.clear();

    executionInputTensors_.reserve(
        inputBuffers_.size()
    );

    executionOutputTensors_.reserve(
        outputBuffers_.size()
    );

    for (const auto& buffer :
         inputBuffers_) {

        if (buffer == nullptr ||
            !buffer->ready()) {

            lastError_ =
                "FaceEmbedding execution input "
                "buffer is not ready";

            return false;
        }

        executionInputTensors_.push_back(
            buffer->tensor()
        );
    }

    for (const auto& buffer :
         outputBuffers_) {

        if (buffer == nullptr ||
            !buffer->ready()) {

            lastError_ =
                "FaceEmbedding execution output "
                "buffer is not ready";

            return false;
        }

        executionOutputTensors_.push_back(
            buffer->tensor()
        );
    }

    if (executionInputTensors_.size() != 1) {

        lastError_ =
            "FaceEmbedding execution input count must be 1";

        return false;
    }

    if (executionOutputTensors_.size() != 1) {

        lastError_ =
            "FaceEmbedding execution output count must be 1";

        return false;
    }

    return true;
}

// =========================================================
// STEP 2
//
// Resize -> RGB -> normalize [-1,1] -> runtime input
// =========================================================

bool FaceEmbeddingModel::preprocess(
    const cv::Mat& bgrFace
)
{
    if (!ready()) {

        lastError_ =
            "FaceEmbeddingModel is not initialized";

        return false;
    }

    if (bgrFace.empty()) {

        lastError_ =
            "FaceEmbedding input image is empty";

        return false;
    }

    if (bgrFace.type() !=
        CV_8UC3) {

        lastError_ =
            "FaceEmbeddingModel expects CV_8UC3 BGR image";

        return false;
    }

    preprocessInfo_.originalWidth =
        bgrFace.cols;

    preprocessInfo_.originalHeight =
        bgrFace.rows;

    preprocessInfo_.resizedWidth =
        INPUT_WIDTH;

    preprocessInfo_.resizedHeight =
        INPUT_HEIGHT;

    // =====================================================
    // Resize to 112x112
    //
    // Python smoke-test assumes face is already cropped
    // and aligned.
    // =====================================================

    cv::Mat resized;

    cv::resize(
        bgrFace,
        resized,
        cv::Size(
            INPUT_WIDTH,
            INPUT_HEIGHT
        ),
        0.0,
        0.0,
        cv::INTER_LINEAR
    );

    // =====================================================
    // BGR -> RGB
    // =====================================================

    cv::Mat rgb;

    cv::cvtColor(
        resized,
        rgb,
        cv::COLOR_BGR2RGB
    );

    inference::QnnTensorBuffer& input =
        *inputBuffers_[0];

    const Qnn_DataType_t inputType =
        dataType(
            input.tensor()
        );

    // =====================================================
    // Normalization:
    //
    // ToTensor()
    // Normalize(mean=0.5, std=0.5)
    //
    // =
    //
    // (pixel / 255 - 0.5) / 0.5
    //
    // =
    //
    // pixel / 127.5 - 1
    // =====================================================

    if (inputType ==
        QNN_DATATYPE_FLOAT_32) {

        auto* destination =
            static_cast<float*>(
                input.data()
            );

        if (destination == nullptr) {

            lastError_ =
                "FaceEmbedding FLOAT32 input buffer is null";

            return false;
        }

        uint64_t index =
            0;

        for (int y = 0;
             y < INPUT_HEIGHT;
             ++y) {

            const cv::Vec3b* row =
                rgb.ptr<cv::Vec3b>(
                    y
                );

            for (int x = 0;
                 x < INPUT_WIDTH;
                 ++x) {

                for (int channel = 0;
                     channel < INPUT_CHANNELS;
                     ++channel) {

                    const float pixel =
                        static_cast<float>(
                            row[x][channel]
                        );

                    destination[index++] =
                        pixel / 127.5F
                        -
                        1.0F;
                }
            }
        }

        if (index !=
            input.elementCount()) {

            lastError_ =
                "FaceEmbedding FLOAT32 preprocess "
                "element count mismatch";

            return false;
        }
    }

    // =====================================================
    // Native quantized input
    // =====================================================

    else if (
        inputType ==
        QNN_DATATYPE_UFIXED_POINT_16
    ) {

        const Qnn_QuantizeParams_t* params =
            quantization(
                input.tensor()
            );

        if (params == nullptr ||
            params->quantizationEncoding !=
                QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {

            lastError_ =
                "FaceEmbedding UFIXED16 input must use "
                "SCALE_OFFSET quantization";

            return false;
        }

        const float quantScale =
            params
                ->scaleOffsetEncoding
                .scale;

        const int32_t quantOffset =
            params
                ->scaleOffsetEncoding
                .offset;

        if (!std::isfinite(
                quantScale
            ) ||
            quantScale <= 0.0F) {

            lastError_ =
                "FaceEmbedding input quantization "
                "scale is invalid";

            return false;
        }

        auto* destination =
            static_cast<uint16_t*>(
                input.data()
            );

        if (destination == nullptr) {

            lastError_ =
                "FaceEmbedding UFIXED16 input buffer is null";

            return false;
        }

        uint64_t index =
            0;

        for (int y = 0;
             y < INPUT_HEIGHT;
             ++y) {

            const cv::Vec3b* row =
                rgb.ptr<cv::Vec3b>(
                    y
                );

            for (int x = 0;
                 x < INPUT_WIDTH;
                 ++x) {

                for (int channel = 0;
                     channel < INPUT_CHANNELS;
                     ++channel) {

                    const float pixel =
                        static_cast<float>(
                            row[x][channel]
                        );

                    const float normalized =
                        pixel / 127.5F
                        -
                        1.0F;

                    destination[index++] =
                        inference::
                        quantizeScaleOffsetToU16(
                            normalized,
                            quantScale,
                            quantOffset
                        );
                }
            }
        }

        if (index !=
            input.elementCount()) {

            lastError_ =
                "FaceEmbedding UFIXED16 preprocess "
                "element count mismatch";

            return false;
        }
    }

    else {

        lastError_ =
            "Unsupported FaceEmbedding input datatype";

        return false;
    }

    lastError_.clear();

    return true;
}

// =========================================================
// STEP 3
//
// HTP inference
// =========================================================

bool FaceEmbeddingModel::infer()
{
    if (!ready()) {

        lastError_ =
            "FaceEmbeddingModel is not ready for inference";

        return false;
    }

    if (executionInputTensors_.size() != 1 ||
        executionOutputTensors_.size() != 1) {

        lastError_ =
            "FaceEmbedding execution tensor count is invalid";

        return false;
    }

    if (!model_.executeGraph(
            0,

            executionInputTensors_.data(),
            static_cast<uint32_t>(
                executionInputTensors_.size()
            ),

            executionOutputTensors_.data(),
            static_cast<uint32_t>(
                executionOutputTensors_.size()
            )
        )) {

        lastError_ =
            "FaceEmbedding graph execution failed: "
            +
            model_.lastError();

        return false;
    }

    lastError_.clear();

    return true;
}

// =========================================================
// STEP 4
//
// Read/dequantize 512-D output
// + L2 normalize
// =========================================================

bool FaceEmbeddingModel::postprocess(
    FaceEmbeddingResult& result
)
{
    result = {};

    if (!ready()) {

        lastError_ =
            "FaceEmbeddingModel is not ready";

        return false;
    }

    const auto* output =
        outputBuffer();

    if (output == nullptr) {

        lastError_ =
            "FaceEmbedding output buffer is null";

        return false;
    }

    if (output->elementCount() !=
        EMBEDDING_DIM) {

        lastError_ =
            "FaceEmbedding output dimension is not 512";

        return false;
    }

    double squaredNorm =
        0.0;

    // =====================================================
    // Read raw/dequantized output
    // =====================================================

    for (uint64_t i = 0;
         i < EMBEDDING_DIM;
         ++i) {

        float value =
            0.0F;

        if (!readTensorValue(
                *output,
                i,
                value
            )) {

            lastError_ =
                "Failed reading FaceEmbedding output";

            return false;
        }

        if (!std::isfinite(
                value
            )) {

            lastError_ =
                "FaceEmbedding output contains "
                "non-finite value";

            return false;
        }

        result.values[i] =
            value;

        squaredNorm +=
            static_cast<double>(
                value
            )
            *
            static_cast<double>(
                value
            );
    }

    const float norm =
        static_cast<float>(
            std::sqrt(
                squaredNorm
            )
        );

    if (!std::isfinite(
            norm
        ) ||
        norm < L2_EPSILON) {

        std::ostringstream oss;

        oss
            << "FaceEmbedding L2 norm is too small: "
            << norm;

        lastError_ =
            oss.str();

        return false;
    }

    result.rawL2Norm =
        norm;

    // =====================================================
    // L2 normalization
    //
    // embedding = embedding / ||embedding||_2
    // =====================================================

    for (float& value :
         result.values) {

        value /=
            norm;
    }

    lastError_.clear();

    return true;
}

// =========================================================
// STEP 5
//
// Complete public API
// =========================================================

bool FaceEmbeddingModel::extract(
    const cv::Mat& bgrFace,
    FaceEmbeddingResult& result
)
{
    result = {};

    if (!preprocess(
            bgrFace
        )) {

        return false;
    }

    if (!infer()) {

        return false;
    }

    if (!postprocess(
            result
        )) {

        return false;
    }

    lastError_.clear();

    return true;
}

// =========================================================
// Tensor reading
// =========================================================

bool FaceEmbeddingModel::readTensorValue(
    const inference::QnnTensorBuffer& buffer,
    uint64_t index,
    float& value
) noexcept
{
    if (index >=
        buffer.elementCount()) {

        return false;
    }

    const Qnn_DataType_t type =
        dataType(
            buffer.tensor()
        );

    // =====================================================
    // FLOAT32
    // =====================================================

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

    // =====================================================
    // Native UFIXED_POINT_16
    //
    // real =
    //     scale * (quantized + offset)
    // =====================================================

    if (type ==
        QNN_DATATYPE_UFIXED_POINT_16) {

        const auto* raw =
            static_cast<const uint16_t*>(
                buffer.data()
            );

        if (raw == nullptr) {

            return false;
        }

        const Qnn_QuantizeParams_t* params =
            quantization(
                buffer.tensor()
            );

        if (params == nullptr ||
            params->quantizationEncoding !=
                QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {

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

        if (!std::isfinite(
                scale
            ) ||
            scale <= 0.0F) {

            return false;
        }

        value =
            inference::dequantizeScaleOffset(
                raw[index],
                scale,
                offset
            );

        return true;
    }

    return false;
}

// =========================================================
// Lifecycle
// =========================================================

void FaceEmbeddingModel::shutdown()
{
    executionInputTensors_.clear();

    executionOutputTensors_.clear();

    inputBuffers_.clear();

    outputBuffers_.clear();

    preprocessInfo_ = {};

    model_.shutdown();
}

bool FaceEmbeddingModel::ready() const noexcept
{
    return
        model_.graphsFinalized()
        &&
        inputBuffers_.size() == 1
        &&
        outputBuffers_.size() == 1
        &&
        executionInputTensors_.size() == 1
        &&
        executionOutputTensors_.size() == 1;
}

const FaceEmbeddingPreprocessInfo&
FaceEmbeddingModel::preprocessInfo() const noexcept
{
    return
        preprocessInfo_;
}

const inference::QnnTensorBuffer*
FaceEmbeddingModel::inputBuffer() const noexcept
{
    if (inputBuffers_.empty()) {

        return nullptr;
    }

    return
        inputBuffers_[0].get();
}

const inference::QnnTensorBuffer*
FaceEmbeddingModel::outputBuffer() const noexcept
{
    if (outputBuffers_.empty()) {

        return nullptr;
    }

    return
        outputBuffers_[0].get();
}

const std::string&
FaceEmbeddingModel::lastError() const noexcept
{
    return
        lastError_;
}

// =========================================================
// QNN tensor helpers
// =========================================================

Qnn_DataType_t
FaceEmbeddingModel::dataType(
    const Qnn_Tensor_t& tensor
) noexcept
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

uint32_t
FaceEmbeddingModel::rank(
    const Qnn_Tensor_t& tensor
) noexcept
{
    switch (tensor.version) {

    case QNN_TENSOR_VERSION_1:

        return
            tensor.v1.rank;

    case QNN_TENSOR_VERSION_2:

        return
            tensor.v2.rank;

    default:

        return
            0;
    }
}

const uint32_t*
FaceEmbeddingModel::dimensions(
    const Qnn_Tensor_t& tensor
) noexcept
{
    switch (tensor.version) {

    case QNN_TENSOR_VERSION_1:

        return
            tensor.v1.dimensions;

    case QNN_TENSOR_VERSION_2:

        return
            tensor.v2.dimensions;

    default:

        return
            nullptr;
    }
}

const Qnn_QuantizeParams_t*
FaceEmbeddingModel::quantization(
    const Qnn_Tensor_t& tensor
) noexcept
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

} // namespace models