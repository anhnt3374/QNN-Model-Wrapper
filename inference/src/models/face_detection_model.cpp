#include "models/face_detection_model.hpp"

#include "inference/qnn_quantization.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdint>
#include <sstream>

namespace models {

FaceDetectionModel::FaceDetectionModel(
    inference::QnnBackend& backend
) noexcept
    : model_(backend)
{
}

FaceDetectionModel::~FaceDetectionModel()
{
    shutdown();
}

bool FaceDetectionModel::initialize(
    const std::string& modelPath
)
{
    shutdown();

    lastError_.clear();

    // =====================================================
    // Load generated SCRFD model
    // =====================================================

    if (!model_.load(
            modelPath
        )) {

        lastError_ =
            "Failed to load face detection model: "
            + model_.lastError();

        return false;
    }

    // =====================================================
    // Compose graph
    // =====================================================

    if (!model_.composeGraphs()) {

        lastError_ =
            "Failed to compose face detection graph: "
            + model_.lastError();

        shutdown();

        return false;
    }

    if (model_.graphCount() != 1) {

        std::ostringstream oss;

        oss
            << "SCRFD expected exactly 1 graph, got "
            << model_.graphCount();

        lastError_ = oss.str();

        shutdown();

        return false;
    }

    // =====================================================
    // Finalize graph
    // =====================================================

    if (!model_.finalizeGraphs()) {

        lastError_ =
            "Failed to finalize face detection graph: "
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
            "SCRFD graph metadata is null";

        shutdown();

        return false;
    }

    // =====================================================
    // Validate expected SCRFD graph contract
    // =====================================================

    if (graph->numInputTensors != 1) {

        std::ostringstream oss;

        oss
            << "SCRFD expected 1 input tensor, got "
            << graph->numInputTensors;

        lastError_ = oss.str();

        shutdown();

        return false;
    }

    if (graph->numOutputTensors != 9) {

        std::ostringstream oss;

        oss
            << "SCRFD expected 9 output tensors, got "
            << graph->numOutputTensors;

        lastError_ = oss.str();

        shutdown();

        return false;
    }

    // =====================================================
    // Allocate persistent I/O buffers
    // =====================================================

    if (!allocateRuntimeBuffers()) {

        shutdown();

        return false;
    }

    // =====================================================
    // Validate SCRFD input metadata
    // =====================================================

    if (!validateScrfdInput()) {

        shutdown();

        return false;
    }

    // =====================================================
    // Build persistent contiguous Qnn_Tensor_t arrays
    //
    // This happens once during initialization.
    //
    // infer() will reuse these arrays every frame.
    // =====================================================

    if (!buildExecutionTensorArrays()) {

        shutdown();

        return false;
    }

    lastError_.clear();

    return true;
}

bool FaceDetectionModel::allocateRuntimeBuffers()
{
    const auto* graph =
        model_.graphInfo(
            0
        );

    if (graph == nullptr) {

        lastError_ =
            "Cannot allocate buffers: graph is null";

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
                << "Failed to allocate input["
                << i
                << "]: "
                << buffer->lastError();

            lastError_ = oss.str();

            return false;
        }

        inputBuffers_.push_back(
            std::move(buffer)
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
                << "Failed to allocate output["
                << i
                << "]: "
                << buffer->lastError();

            lastError_ = oss.str();

            return false;
        }

        outputBuffers_.push_back(
            std::move(buffer)
        );
    }

    return true;
}

bool FaceDetectionModel::buildExecutionTensorArrays()
{
    executionInputTensors_.clear();
    executionOutputTensors_.clear();

    executionInputTensors_.reserve(
        inputBuffers_.size()
    );

    executionOutputTensors_.reserve(
        outputBuffers_.size()
    );

    // =====================================================
    // Each copied Qnn_Tensor_t keeps clientBuf.data pointing
    // to memory owned by the corresponding QnnTensorBuffer.
    //
    // QnnTensorBuffer objects themselves are heap-allocated
    // and remain stable behind unique_ptr.
    // =====================================================

    for (const auto& buffer :
         inputBuffers_) {

        if (buffer == nullptr ||
            !buffer->ready()) {

            lastError_ =
                "Cannot build execution input tensors: "
                "runtime buffer is not ready";

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
                "Cannot build execution output tensors: "
                "runtime buffer is not ready";

            return false;
        }

        executionOutputTensors_.push_back(
            buffer->tensor()
        );
    }

    if (executionInputTensors_.size() !=
        inputBuffers_.size()) {

        lastError_ =
            "Execution input tensor count mismatch";

        return false;
    }

    if (executionOutputTensors_.size() !=
        outputBuffers_.size()) {

        lastError_ =
            "Execution output tensor count mismatch";

        return false;
    }

    return true;
}

bool FaceDetectionModel::validateScrfdInput()
{
    if (inputBuffers_.size() != 1) {

        lastError_ =
            "SCRFD runtime input buffer count is not 1";

        return false;
    }

    const Qnn_Tensor_t& tensor =
        inputBuffers_[0]->tensor();

    // =====================================================
    // Datatype
    // =====================================================

    if (dataType(
            tensor
        ) !=
        QNN_DATATYPE_UFIXED_POINT_16) {

        lastError_ =
            "SCRFD input must be "
            "QNN_DATATYPE_UFIXED_POINT_16";

        return false;
    }

    // =====================================================
    // Shape [1,640,640,3]
    // =====================================================

    if (rank(
            tensor
        ) != 4) {

        lastError_ =
            "SCRFD input rank must be 4";

        return false;
    }

    const uint32_t* dims =
        dimensions(
            tensor
        );

    if (dims == nullptr) {

        lastError_ =
            "SCRFD input dimensions are null";

        return false;
    }

    if (dims[0] != 1 ||
        dims[1] != INPUT_HEIGHT ||
        dims[2] != INPUT_WIDTH ||
        dims[3] != INPUT_CHANNELS) {

        std::ostringstream oss;

        oss
            << "Unexpected SCRFD input shape: ["
            << dims[0]
            << ", "
            << dims[1]
            << ", "
            << dims[2]
            << ", "
            << dims[3]
            << "]";

        lastError_ = oss.str();

        return false;
    }

    // =====================================================
    // Quantization
    // =====================================================

    const Qnn_QuantizeParams_t* params =
        quantization(
            tensor
        );

    if (params == nullptr) {

        lastError_ =
            "SCRFD input quantization metadata is null";

        return false;
    }

    if (params->quantizationEncoding !=
        QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {

        lastError_ =
            "SCRFD input must use SCALE_OFFSET quantization";

        return false;
    }

    if (!std::isfinite(
            params
                ->scaleOffsetEncoding
                .scale
        ) ||
        params
            ->scaleOffsetEncoding
            .scale <= 0.0F) {

        lastError_ =
            "SCRFD input quantization scale is invalid";

        return false;
    }

    return true;
}

bool FaceDetectionModel::preprocess(
    const cv::Mat& bgrImage
)
{
    if (!ready()) {

        lastError_ =
            "FaceDetectionModel is not initialized";

        return false;
    }

    if (bgrImage.empty()) {

        lastError_ =
            "Input image is empty";

        return false;
    }

    if (bgrImage.type() != CV_8UC3) {

        lastError_ =
            "FaceDetectionModel expects CV_8UC3 BGR image";

        return false;
    }

    const int originalWidth =
        bgrImage.cols;

    const int originalHeight =
        bgrImage.rows;

    if (originalWidth <= 0 ||
        originalHeight <= 0) {

        lastError_ =
            "Input image dimensions are invalid";

        return false;
    }

    // =====================================================
    // Same resize logic as the working Python preprocess:
    //
    // keep aspect ratio
    // place resized image at top-left
    // =====================================================

    const double imageRatio =
        static_cast<double>(
            originalHeight
        )
        /
        static_cast<double>(
            originalWidth
        );

    const double modelRatio =
        static_cast<double>(
            INPUT_HEIGHT
        )
        /
        static_cast<double>(
            INPUT_WIDTH
        );

    int resizedWidth = 0;
    int resizedHeight = 0;

    if (imageRatio > modelRatio) {

        resizedHeight =
            INPUT_HEIGHT;

        resizedWidth =
            static_cast<int>(
                static_cast<double>(
                    resizedHeight
                )
                /
                imageRatio
            );

    } else {

        resizedWidth =
            INPUT_WIDTH;

        resizedHeight =
            static_cast<int>(
                static_cast<double>(
                    resizedWidth
                )
                *
                imageRatio
            );
    }

    if (resizedWidth <= 0 ||
        resizedHeight <= 0) {

        lastError_ =
            "Calculated SCRFD resize dimensions are invalid";

        return false;
    }

    const float detScale =
        static_cast<float>(
            resizedHeight
        )
        /
        static_cast<float>(
            originalHeight
        );

    // =====================================================
    // Resize
    // =====================================================

    cv::Mat resized;

    cv::resize(
        bgrImage,
        resized,
        cv::Size(
            resizedWidth,
            resizedHeight
        ),
        0.0,
        0.0,
        cv::INTER_LINEAR
    );

    // =====================================================
    // Top-left letterbox
    // =====================================================

    cv::Mat canvas =
        cv::Mat::zeros(
            INPUT_HEIGHT,
            INPUT_WIDTH,
            CV_8UC3
        );

    resized.copyTo(
        canvas(
            cv::Rect(
                0,
                0,
                resizedWidth,
                resizedHeight
            )
        )
    );

    // =====================================================
    // OpenCV BGR -> RGB
    // =====================================================

    cv::Mat rgb;

    cv::cvtColor(
        canvas,
        rgb,
        cv::COLOR_BGR2RGB
    );

    // =====================================================
    // QNN input metadata
    // =====================================================

    inference::QnnTensorBuffer&
        input =
            *inputBuffers_[0];

    const Qnn_QuantizeParams_t* params =
        quantization(
            input.tensor()
        );

    if (params == nullptr) {

        lastError_ =
            "SCRFD input quantization metadata disappeared";

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

    auto* destination =
        static_cast<uint16_t*>(
            input.data()
        );

    if (destination == nullptr) {

        lastError_ =
            "SCRFD input runtime buffer is null";

        return false;
    }

    // =====================================================
    // Python reference:
    //
    // (rgb.astype(float32) - 127.5) / 128.0
    //
    // Then convert normalized float to QNN UFIXED_POINT_16.
    // =====================================================

    uint64_t destinationIndex = 0;

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
                    (
                        pixel
                        -
                        127.5F
                    )
                    /
                    128.0F;

                destination[
                    destinationIndex++
                ] =
                    inference::
                    quantizeScaleOffsetToU16(
                        normalized,
                        scale,
                        offset
                    );
            }
        }
    }

    if (destinationIndex !=
        input.elementCount()) {

        std::ostringstream oss;

        oss
            << "SCRFD preprocess element count mismatch. "
            << "written="
            << destinationIndex
            << ", expected="
            << input.elementCount();

        lastError_ = oss.str();

        return false;
    }

    // =====================================================
    // Save values required later by SCRFD postprocessing.
    // =====================================================

    preprocessInfo_.originalWidth =
        originalWidth;

    preprocessInfo_.originalHeight =
        originalHeight;

    preprocessInfo_.resizedWidth =
        resizedWidth;

    preprocessInfo_.resizedHeight =
        resizedHeight;

    preprocessInfo_.detScale =
        detScale;

    lastError_.clear();

    return true;
}

bool FaceDetectionModel::infer()
{
    if (!ready()) {

        lastError_ =
            "FaceDetectionModel is not ready for inference";

        return false;
    }

    if (executionInputTensors_.size() != 1) {

        lastError_ =
            "SCRFD execution input count is invalid";

        return false;
    }

    if (executionOutputTensors_.size() != 9) {

        lastError_ =
            "SCRFD execution output count is invalid";

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
            "SCRFD graph execution failed: "
            + model_.lastError();

        return false;
    }

    lastError_.clear();

    return true;
}

void FaceDetectionModel::shutdown()
{
    // Tensor copies contain pointers to QnnTensorBuffer-owned
    // memory, therefore clear execution views first.

    executionInputTensors_.clear();

    executionOutputTensors_.clear();

    inputBuffers_.clear();

    outputBuffers_.clear();

    preprocessInfo_ = {};

    model_.shutdown();
}

bool FaceDetectionModel::ready() const noexcept
{
    return
        model_.graphsFinalized()
        &&
        inputBuffers_.size() == 1
        &&
        outputBuffers_.size() == 9
        &&
        executionInputTensors_.size() == 1
        &&
        executionOutputTensors_.size() == 9;
}

const FaceDetectionPreprocessInfo&
FaceDetectionModel::preprocessInfo() const noexcept
{
    return preprocessInfo_;
}

const inference::QnnTensorBuffer*
FaceDetectionModel::inputBuffer() const noexcept
{
    if (inputBuffers_.empty()) {
        return nullptr;
    }

    return
        inputBuffers_[0].get();
}

std::size_t
FaceDetectionModel::outputCount() const noexcept
{
    return outputBuffers_.size();
}

const inference::QnnTensorBuffer*
FaceDetectionModel::outputBuffer(
    std::size_t index
) const noexcept
{
    if (index >=
        outputBuffers_.size()) {

        return nullptr;
    }

    return
        outputBuffers_[index].get();
}

const std::string&
FaceDetectionModel::lastError() const noexcept
{
    return lastError_;
}

Qnn_DataType_t FaceDetectionModel::dataType(
    const Qnn_Tensor_t& tensor
) noexcept
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

uint32_t FaceDetectionModel::rank(
    const Qnn_Tensor_t& tensor
) noexcept
{
    switch (tensor.version) {

    case QNN_TENSOR_VERSION_1:
        return tensor.v1.rank;

    case QNN_TENSOR_VERSION_2:
        return tensor.v2.rank;

    default:
        return 0;
    }
}

const uint32_t*
FaceDetectionModel::dimensions(
    const Qnn_Tensor_t& tensor
) noexcept
{
    switch (tensor.version) {

    case QNN_TENSOR_VERSION_1:
        return tensor.v1.dimensions;

    case QNN_TENSOR_VERSION_2:
        return tensor.v2.dimensions;

    default:
        return nullptr;
    }
}

const Qnn_QuantizeParams_t*
FaceDetectionModel::quantization(
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
        return nullptr;
    }
}

} // namespace models