#include "models/face_detection_model.hpp"

#include "inference/qnn_quantization.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
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

    if (!model_.load(
            modelPath
        )) {

        lastError_ =
            "Failed to load face detection model: "
            + model_.lastError();

        return false;
    }

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

    if (!allocateRuntimeBuffers()) {

        shutdown();

        return false;
    }

    if (!validateScrfdInput()) {

        shutdown();

        return false;
    }

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

    if (dataType(
            tensor
        ) !=
        QNN_DATATYPE_UFIXED_POINT_16) {

        lastError_ =
            "SCRFD input must be "
            "QNN_DATATYPE_UFIXED_POINT_16";

        return false;
    }

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

    cv::Mat rgb;

    cv::cvtColor(
        canvas,
        rgb,
        cv::COLOR_BGR2RGB
    );

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

bool FaceDetectionModel::decodeStride8(
    std::vector<FaceDetectionProposal>& proposals,
    float scoreThreshold
)
{
    proposals.clear();

    if (!decodeLevel(
            8,
            "score_8",
            "bbox_8",
            "kps_8",
            scoreThreshold,
            proposals
        )) {

        return false;
    }

    lastError_.clear();

    return true;
}

bool FaceDetectionModel::decodeAll(
    std::vector<FaceDetectionProposal>& proposals,
    float scoreThreshold
)
{
    proposals.clear();

    if (!std::isfinite(
            scoreThreshold
        )) {

        lastError_ =
            "Score threshold is not finite";

        return false;
    }

    // Maximum possible number before threshold:
    //
    // stride 8  = 12800
    // stride 16 = 3200
    // stride 32 = 800
    //
    // total     = 16800
    proposals.reserve(
        16800
    );

    if (!decodeLevel(
            8,
            "score_8",
            "bbox_8",
            "kps_8",
            scoreThreshold,
            proposals
        )) {

        return false;
    }

    if (!decodeLevel(
            16,
            "score_16",
            "bbox_16",
            "kps_16",
            scoreThreshold,
            proposals
        )) {

        return false;
    }

    if (!decodeLevel(
            32,
            "score_32",
            "bbox_32",
            "kps_32",
            scoreThreshold,
            proposals
        )) {

        return false;
    }

    // Same ordering as Python:
    //
    // order = scores.argsort()[::-1]
    std::sort(
        proposals.begin(),
        proposals.end(),
        [](
            const FaceDetectionProposal& lhs,
            const FaceDetectionProposal& rhs
        ) {
            return lhs.score > rhs.score;
        }
    );

    lastError_.clear();

    return true;
}

bool FaceDetectionModel::decodeLevel(
    int stride,
    const char* scoreTensorName,
    const char* bboxTensorName,
    const char* kpsTensorName,
    float scoreThreshold,
    std::vector<FaceDetectionProposal>& proposals
)
{
    if (!ready()) {

        lastError_ =
            "FaceDetectionModel is not ready";

        return false;
    }

    if (!std::isfinite(
            scoreThreshold
        )) {

        lastError_ =
            "Score threshold is not finite";

        return false;
    }

    if (stride <= 0 ||
        INPUT_WIDTH % stride != 0 ||
        INPUT_HEIGHT % stride != 0) {

        lastError_ =
            "Invalid SCRFD stride";

        return false;
    }

    const inference::QnnTensorBuffer*
        scoreBuffer =
            outputBufferByName(
                scoreTensorName
            );

    const inference::QnnTensorBuffer*
        bboxBuffer =
            outputBufferByName(
                bboxTensorName
            );

    const inference::QnnTensorBuffer*
        kpsBuffer =
            outputBufferByName(
                kpsTensorName
            );

    if (scoreBuffer == nullptr ||
        bboxBuffer == nullptr ||
        kpsBuffer == nullptr) {

        std::ostringstream oss;

        oss
            << "SCRFD output tensors missing for stride "
            << stride;

        lastError_ = oss.str();

        return false;
    }

    if (dataType(
            scoreBuffer->tensor()
        ) !=
            QNN_DATATYPE_UFIXED_POINT_16 ||
        dataType(
            bboxBuffer->tensor()
        ) !=
            QNN_DATATYPE_UFIXED_POINT_16 ||
        dataType(
            kpsBuffer->tensor()
        ) !=
            QNN_DATATYPE_UFIXED_POINT_16) {

        std::ostringstream oss;

        oss
            << "SCRFD stride "
            << stride
            << " outputs must be UFIXED_POINT_16";

        lastError_ = oss.str();

        return false;
    }

    const uint64_t gridWidth =
        static_cast<uint64_t>(
            INPUT_WIDTH / stride
        );

    const uint64_t gridHeight =
        static_cast<uint64_t>(
            INPUT_HEIGHT / stride
        );

    const uint64_t candidateCount =
        gridWidth
        *
        gridHeight
        *
        NUM_ANCHORS;

    if (scoreBuffer->elementCount() !=
        candidateCount) {

        std::ostringstream oss;

        oss
            << scoreTensorName
            << " element count mismatch. expected="
            << candidateCount
            << ", actual="
            << scoreBuffer->elementCount();

        lastError_ = oss.str();

        return false;
    }

    if (bboxBuffer->elementCount() !=
        candidateCount * 4) {

        std::ostringstream oss;

        oss
            << bboxTensorName
            << " element count mismatch. expected="
            << candidateCount * 4
            << ", actual="
            << bboxBuffer->elementCount();

        lastError_ = oss.str();

        return false;
    }

    if (kpsBuffer->elementCount() !=
        candidateCount * 10) {

        std::ostringstream oss;

        oss
            << kpsTensorName
            << " element count mismatch. expected="
            << candidateCount * 10
            << ", actual="
            << kpsBuffer->elementCount();

        lastError_ = oss.str();

        return false;
    }

    const Qnn_QuantizeParams_t*
        scoreQuant =
            quantization(
                scoreBuffer->tensor()
            );

    const Qnn_QuantizeParams_t*
        bboxQuant =
            quantization(
                bboxBuffer->tensor()
            );

    const Qnn_QuantizeParams_t*
        kpsQuant =
            quantization(
                kpsBuffer->tensor()
            );

    if (scoreQuant == nullptr ||
        bboxQuant == nullptr ||
        kpsQuant == nullptr) {

        std::ostringstream oss;

        oss
            << "SCRFD stride "
            << stride
            << " quantization metadata is missing";

        lastError_ = oss.str();

        return false;
    }

    if (scoreQuant->quantizationEncoding !=
            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET ||
        bboxQuant->quantizationEncoding !=
            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET ||
        kpsQuant->quantizationEncoding !=
            QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {

        std::ostringstream oss;

        oss
            << "SCRFD stride "
            << stride
            << " outputs must use SCALE_OFFSET";

        lastError_ = oss.str();

        return false;
    }

    const float scoreScale =
        scoreQuant
            ->scaleOffsetEncoding
            .scale;

    const int32_t scoreOffset =
        scoreQuant
            ->scaleOffsetEncoding
            .offset;

    const float bboxScale =
        bboxQuant
            ->scaleOffsetEncoding
            .scale;

    const int32_t bboxOffset =
        bboxQuant
            ->scaleOffsetEncoding
            .offset;

    const float kpsScale =
        kpsQuant
            ->scaleOffsetEncoding
            .scale;

    const int32_t kpsOffset =
        kpsQuant
            ->scaleOffsetEncoding
            .offset;

    if (!std::isfinite(scoreScale) ||
        scoreScale <= 0.0F ||
        !std::isfinite(bboxScale) ||
        bboxScale <= 0.0F ||
        !std::isfinite(kpsScale) ||
        kpsScale <= 0.0F) {

        std::ostringstream oss;

        oss
            << "SCRFD stride "
            << stride
            << " contains invalid quantization scale";

        lastError_ = oss.str();

        return false;
    }

    const auto* scoreRaw =
        static_cast<const uint16_t*>(
            scoreBuffer->data()
        );

    const auto* bboxRaw =
        static_cast<const uint16_t*>(
            bboxBuffer->data()
        );

    const auto* kpsRaw =
        static_cast<const uint16_t*>(
            kpsBuffer->data()
        );

    if (scoreRaw == nullptr ||
        bboxRaw == nullptr ||
        kpsRaw == nullptr) {

        std::ostringstream oss;

        oss
            << "SCRFD stride "
            << stride
            << " runtime buffer is null";

        lastError_ = oss.str();

        return false;
    }

    for (uint64_t candidateIndex = 0;
         candidateIndex < candidateCount;
         ++candidateIndex) {

        // =================================================
        // Score
        // =================================================

        const float score =
            inference::dequantizeScaleOffset(
                scoreRaw[
                    candidateIndex
                ],
                scoreScale,
                scoreOffset
            );

        if (score <
            scoreThreshold) {

            continue;
        }

        // =================================================
        // Anchor center
        //
        // Python:
        //
        // centers = [xx, yy] * stride
        // repeat each center twice.
        // =================================================

        const uint64_t gridIndex =
            candidateIndex
            /
            NUM_ANCHORS;

        const uint64_t gridX =
            gridIndex
            %
            gridWidth;

        const uint64_t gridY =
            gridIndex
            /
            gridWidth;

        const float centerX =
            static_cast<float>(
                gridX
            )
            *
            static_cast<float>(
                stride
            );

        const float centerY =
            static_cast<float>(
                gridY
            )
            *
            static_cast<float>(
                stride
            );

        // =================================================
        // BBox
        // =================================================

        const uint64_t bboxBase =
            candidateIndex * 4;

        const float left =
            inference::dequantizeScaleOffset(
                bboxRaw[
                    bboxBase + 0
                ],
                bboxScale,
                bboxOffset
            )
            *
            static_cast<float>(
                stride
            );

        const float top =
            inference::dequantizeScaleOffset(
                bboxRaw[
                    bboxBase + 1
                ],
                bboxScale,
                bboxOffset
            )
            *
            static_cast<float>(
                stride
            );

        const float right =
            inference::dequantizeScaleOffset(
                bboxRaw[
                    bboxBase + 2
                ],
                bboxScale,
                bboxOffset
            )
            *
            static_cast<float>(
                stride
            );

        const float bottom =
            inference::dequantizeScaleOffset(
                bboxRaw[
                    bboxBase + 3
                ],
                bboxScale,
                bboxOffset
            )
            *
            static_cast<float>(
                stride
            );

        FaceDetectionProposal proposal;

        proposal.score =
            score;

        proposal.bbox[0] =
            centerX
            -
            left;

        proposal.bbox[1] =
            centerY
            -
            top;

        proposal.bbox[2] =
            centerX
            +
            right;

        proposal.bbox[3] =
            centerY
            +
            bottom;

        // =================================================
        // 5 keypoints
        // =================================================

        const uint64_t kpsBase =
            candidateIndex * 10;

        for (uint64_t pointIndex = 0;
             pointIndex < 5;
             ++pointIndex) {

            const uint64_t xIndex =
                kpsBase
                +
                pointIndex * 2;

            const uint64_t yIndex =
                xIndex + 1;

            const float relativeX =
                inference::dequantizeScaleOffset(
                    kpsRaw[
                        xIndex
                    ],
                    kpsScale,
                    kpsOffset
                )
                *
                static_cast<float>(
                    stride
                );

            const float relativeY =
                inference::dequantizeScaleOffset(
                    kpsRaw[
                        yIndex
                    ],
                    kpsScale,
                    kpsOffset
                )
                *
                static_cast<float>(
                    stride
                );

            proposal.landmarks[
                pointIndex * 2
            ] =
                centerX
                +
                relativeX;

            proposal.landmarks[
                pointIndex * 2 + 1
            ] =
                centerY
                +
                relativeY;
        }

        proposals.push_back(
            proposal
        );
    }

    return true;
}

const inference::QnnTensorBuffer*
FaceDetectionModel::outputBufferByName(
    const char* name
) const noexcept
{
    if (name == nullptr) {
        return nullptr;
    }

    for (const auto& buffer :
         outputBuffers_) {

        if (buffer == nullptr) {
            continue;
        }

        if (buffer->name() ==
            name) {

            return
                buffer.get();
        }
    }

    return nullptr;
}

void FaceDetectionModel::shutdown()
{
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