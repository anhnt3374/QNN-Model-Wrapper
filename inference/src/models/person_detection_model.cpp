#include "models/person_detection_model.hpp"

#include "inference/qnn_quantization.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace models {

PersonDetectionModel::PersonDetectionModel(
    inference::QnnBackend& backend
) noexcept
    : model_(backend)
{
}

PersonDetectionModel::~PersonDetectionModel()
{
    shutdown();
}

bool PersonDetectionModel::initialize(
    const std::string& modelPath
)
{
    shutdown();

    lastError_.clear();

    if (!model_.load(
            modelPath
        )) {

        lastError_ =
            "Failed to load person detection model: "
            + model_.lastError();

        return false;
    }

    if (!model_.composeGraphs()) {

        lastError_ =
            "Failed to compose person detection graph: "
            + model_.lastError();

        shutdown();

        return false;
    }

    if (model_.graphCount() != 1) {

        std::ostringstream oss;

        oss
            << "Person detector expected exactly 1 graph, got "
            << model_.graphCount();

        lastError_ = oss.str();

        shutdown();

        return false;
    }

    if (!model_.finalizeGraphs()) {

        lastError_ =
            "Failed to finalize person detection graph: "
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
            "Person detector graph metadata is null";

        shutdown();

        return false;
    }

    if (graph->numInputTensors != 1) {

        std::ostringstream oss;

        oss
            << "Person detector expected 1 input tensor, got "
            << graph->numInputTensors;

        lastError_ = oss.str();

        shutdown();

        return false;
    }

    if (graph->numOutputTensors != 2) {

        std::ostringstream oss;

        oss
            << "Person detector expected 2 output tensors, got "
            << graph->numOutputTensors;

        lastError_ = oss.str();

        shutdown();

        return false;
    }

    if (!allocateRuntimeBuffers()) {

        shutdown();

        return false;
    }

    if (!validateModelContract()) {

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

bool PersonDetectionModel::allocateRuntimeBuffers()
{
    const auto* graph =
        model_.graphInfo(
            0
        );

    if (graph == nullptr) {

        lastError_ =
            "Cannot allocate person detector buffers: "
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
                << "Failed to allocate person input["
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
                << "Failed to allocate person output["
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

bool PersonDetectionModel::buildExecutionTensorArrays()
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
                "Cannot build person execution inputs: "
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
                "Cannot build person execution outputs: "
                "runtime buffer is not ready";

            return false;
        }

        executionOutputTensors_.push_back(
            buffer->tensor()
        );
    }

    if (executionInputTensors_.size() != 1) {

        lastError_ =
            "Person execution input count is not 1";

        return false;
    }

    if (executionOutputTensors_.size() != 2) {

        lastError_ =
            "Person execution output count is not 2";

        return false;
    }

    return true;
}

bool PersonDetectionModel::validateModelContract()
{
    if (inputBuffers_.size() != 1) {

        lastError_ =
            "Person detector runtime input count is not 1";

        return false;
    }

    if (outputBuffers_.size() != 2) {

        lastError_ =
            "Person detector runtime output count is not 2";

        return false;
    }

    const auto& input =
        *inputBuffers_[0];

    if (input.name() !=
        "images") {

        std::ostringstream oss;

        oss
            << "Expected person input tensor 'images', got '"
            << input.name()
            << "'";

        lastError_ = oss.str();

        return false;
    }

    const Qnn_Tensor_t& inputTensor =
        input.tensor();

    if (rank(
            inputTensor
        ) != 4) {

        lastError_ =
            "Person input tensor rank must be 4";

        return false;
    }

    const uint32_t* inputDims =
        dimensions(
            inputTensor
        );

    if (inputDims == nullptr) {

        lastError_ =
            "Person input dimensions are null";

        return false;
    }

    if (inputDims[0] != 1 ||
        inputDims[1] != INPUT_HEIGHT ||
        inputDims[2] != INPUT_WIDTH ||
        inputDims[3] != INPUT_CHANNELS) {

        std::ostringstream oss;

        oss
            << "Unexpected person input shape: ["
            << inputDims[0]
            << ", "
            << inputDims[1]
            << ", "
            << inputDims[2]
            << ", "
            << inputDims[3]
            << "]";

        lastError_ = oss.str();

        return false;
    }

    const Qnn_DataType_t inputType =
        dataType(
            inputTensor
        );

    if (inputType !=
            QNN_DATATYPE_FLOAT_32 &&
        inputType !=
            QNN_DATATYPE_UFIXED_POINT_16) {

        lastError_ =
            "Unsupported person input datatype";

        return false;
    }

    const auto* boxes =
        outputBufferByName(
            "boxes_out"
        );

    const auto* conf =
        outputBufferByName(
            "conf_out"
        );

    if (boxes == nullptr) {

        lastError_ =
            "Person detector output 'boxes_out' is missing";

        return false;
    }

    if (conf == nullptr) {

        lastError_ =
            "Person detector output 'conf_out' is missing";

        return false;
    }

    // boxes_out [1,4,8400]

    if (rank(
            boxes->tensor()
        ) != 3) {

        lastError_ =
            "boxes_out rank must be 3";

        return false;
    }

    const uint32_t* boxesDims =
        dimensions(
            boxes->tensor()
        );

    if (boxesDims == nullptr) {

        lastError_ =
            "boxes_out dimensions are null";

        return false;
    }

    if (boxesDims[0] != 1 ||
        boxesDims[1] != 4 ||
        boxesDims[2] != NUM_PREDICTIONS) {

        std::ostringstream oss;

        oss
            << "Unexpected boxes_out shape: ["
            << boxesDims[0]
            << ", "
            << boxesDims[1]
            << ", "
            << boxesDims[2]
            << "]";

        lastError_ = oss.str();

        return false;
    }

    if (boxes->elementCount() !=
        4ULL * NUM_PREDICTIONS) {

        lastError_ =
            "Unexpected boxes_out element count";

        return false;
    }

    // conf_out [1,1,8400]

    if (rank(
            conf->tensor()
        ) != 3) {

        lastError_ =
            "conf_out rank must be 3";

        return false;
    }

    const uint32_t* confDims =
        dimensions(
            conf->tensor()
        );

    if (confDims == nullptr) {

        lastError_ =
            "conf_out dimensions are null";

        return false;
    }

    if (confDims[0] != 1 ||
        confDims[1] != 1 ||
        confDims[2] != NUM_PREDICTIONS) {

        std::ostringstream oss;

        oss
            << "Unexpected conf_out shape: ["
            << confDims[0]
            << ", "
            << confDims[1]
            << ", "
            << confDims[2]
            << "]";

        lastError_ = oss.str();

        return false;
    }

    if (conf->elementCount() !=
        NUM_PREDICTIONS) {

        lastError_ =
            "Unexpected conf_out element count";

        return false;
    }

    const Qnn_DataType_t boxesType =
        dataType(
            boxes->tensor()
        );

    const Qnn_DataType_t confType =
        dataType(
            conf->tensor()
        );

    if (boxesType !=
            QNN_DATATYPE_FLOAT_32 &&
        boxesType !=
            QNN_DATATYPE_UFIXED_POINT_16) {

        lastError_ =
            "Unsupported boxes_out datatype";

        return false;
    }

    if (confType !=
            QNN_DATATYPE_FLOAT_32 &&
        confType !=
            QNN_DATATYPE_UFIXED_POINT_16) {

        lastError_ =
            "Unsupported conf_out datatype";

        return false;
    }

    lastError_.clear();

    return true;
}

bool PersonDetectionModel::preprocess(
    const cv::Mat& bgrImage
)
{
    if (!ready()) {

        lastError_ =
            "PersonDetectionModel is not initialized";

        return false;
    }

    if (bgrImage.empty()) {

        lastError_ =
            "Person detector input image is empty";

        return false;
    }

    if (bgrImage.type() !=
        CV_8UC3) {

        lastError_ =
            "PersonDetectionModel expects CV_8UC3 BGR image";

        return false;
    }

    const int originalWidth =
        bgrImage.cols;

    const int originalHeight =
        bgrImage.rows;

    if (originalWidth <= 0 ||
        originalHeight <= 0) {

        lastError_ =
            "Person detector image dimensions are invalid";

        return false;
    }

    const double scale =
        std::min(
            static_cast<double>(
                INPUT_WIDTH
            )
            /
            static_cast<double>(
                originalWidth
            ),

            static_cast<double>(
                INPUT_HEIGHT
            )
            /
            static_cast<double>(
                originalHeight
            )
        );

    const int resizedWidth =
        pythonRound(
            static_cast<double>(
                originalWidth
            )
            *
            scale
        );

    const int resizedHeight =
        pythonRound(
            static_cast<double>(
                originalHeight
            )
            *
            scale
        );

    if (resizedWidth <= 0 ||
        resizedHeight <= 0 ||
        resizedWidth > INPUT_WIDTH ||
        resizedHeight > INPUT_HEIGHT) {

        lastError_ =
            "Person detector calculated invalid resize";

        return false;
    }

    const double dw =
        (
            static_cast<double>(
                INPUT_WIDTH
            )
            -
            resizedWidth
        )
        /
        2.0;

    const double dh =
        (
            static_cast<double>(
                INPUT_HEIGHT
            )
            -
            resizedHeight
        )
        /
        2.0;

    const int padLeft =
        pythonRound(
            dw - 0.1
        );

    const int padRight =
        pythonRound(
            dw + 0.1
        );

    const int padTop =
        pythonRound(
            dh - 0.1
        );

    const int padBottom =
        pythonRound(
            dh + 0.1
        );

    if (resizedWidth +
            padLeft +
            padRight !=
        INPUT_WIDTH) {

        lastError_ =
            "Person detector horizontal letterbox "
            "does not equal 640";

        return false;
    }

    if (resizedHeight +
            padTop +
            padBottom !=
        INPUT_HEIGHT) {

        lastError_ =
            "Person detector vertical letterbox "
            "does not equal 640";

        return false;
    }

    cv::Mat resized;

    if (resizedWidth !=
            originalWidth ||
        resizedHeight !=
            originalHeight) {

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

    } else {

        resized =
            bgrImage;
    }

    cv::Mat letterboxed;

    cv::copyMakeBorder(
        resized,
        letterboxed,
        padTop,
        padBottom,
        padLeft,
        padRight,
        cv::BORDER_CONSTANT,
        cv::Scalar(
            PAD_VALUE,
            PAD_VALUE,
            PAD_VALUE
        )
    );

    if (letterboxed.cols !=
            INPUT_WIDTH ||
        letterboxed.rows !=
            INPUT_HEIGHT) {

        lastError_ =
            "Person letterbox output is not 640x640";

        return false;
    }

    cv::Mat rgb;

    cv::cvtColor(
        letterboxed,
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
    // FLOAT32 input
    // =====================================================

    if (inputType ==
        QNN_DATATYPE_FLOAT_32) {

        auto* destination =
            static_cast<float*>(
                input.data()
            );

        if (destination == nullptr) {

            lastError_ =
                "Person FLOAT32 input buffer is null";

            return false;
        }

        uint64_t index = 0;

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

                    destination[
                        index++
                    ] =
                        static_cast<float>(
                            row[x][channel]
                        )
                        /
                        255.0F;
                }
            }
        }

        if (index !=
            input.elementCount()) {

            lastError_ =
                "Person FLOAT32 preprocess "
                "element count mismatch";

            return false;
        }
    }

    // =====================================================
    // UFIXED16 input
    // =====================================================

    else if (inputType ==
             QNN_DATATYPE_UFIXED_POINT_16) {

        const Qnn_QuantizeParams_t* params =
            quantization(
                input.tensor()
            );

        if (params == nullptr ||
            params->quantizationEncoding !=
                QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {

            lastError_ =
                "Invalid person input quantization metadata";

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
                "Person input quantization scale is invalid";

            return false;
        }

        auto* destination =
            static_cast<uint16_t*>(
                input.data()
            );

        if (destination == nullptr) {

            lastError_ =
                "Person UFIXED16 input buffer is null";

            return false;
        }

        uint64_t index = 0;

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

                    const float normalized =
                        static_cast<float>(
                            row[x][channel]
                        )
                        /
                        255.0F;

                    destination[
                        index++
                    ] =
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
                "Person UFIXED16 preprocess "
                "element count mismatch";

            return false;
        }
    }

    else {

        lastError_ =
            "Unsupported person input datatype";

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

    preprocessInfo_.scale =
        static_cast<float>(
            scale
        );

    preprocessInfo_.padLeft =
        padLeft;

    preprocessInfo_.padRight =
        padRight;

    preprocessInfo_.padTop =
        padTop;

    preprocessInfo_.padBottom =
        padBottom;

    lastError_.clear();

    return true;
}

bool PersonDetectionModel::infer()
{
    if (!ready()) {

        lastError_ =
            "PersonDetectionModel is not ready for inference";

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
            "Person graph execution failed: "
            + model_.lastError();

        return false;
    }

    lastError_.clear();

    return true;
}

bool PersonDetectionModel::decode(
    std::vector<PersonDetectionProposal>& proposals,
    float confidenceThreshold
)
{
    proposals.clear();

    if (!ready()) {

        lastError_ =
            "PersonDetectionModel is not ready";

        return false;
    }

    if (!std::isfinite(
            confidenceThreshold
        )) {

        lastError_ =
            "Confidence threshold is not finite";

        return false;
    }

    if (!std::isfinite(
            preprocessInfo_.scale
        ) ||
        preprocessInfo_.scale <= 0.0F) {

        lastError_ =
            "Person preprocessing geometry is invalid";

        return false;
    }

    if (preprocessInfo_.originalWidth <= 0 ||
        preprocessInfo_.originalHeight <= 0) {

        lastError_ =
            "Original image dimensions are invalid";

        return false;
    }

    const auto* boxes =
        outputBufferByName(
            "boxes_out"
        );

    const auto* conf =
        outputBufferByName(
            "conf_out"
        );

    if (boxes == nullptr ||
        conf == nullptr) {

        lastError_ =
            "Person detector outputs are missing";

        return false;
    }

    if (boxes->elementCount() !=
            4ULL * NUM_PREDICTIONS ||
        conf->elementCount() !=
            NUM_PREDICTIONS) {

        lastError_ =
            "Person detector output element count mismatch";

        return false;
    }

    constexpr uint64_t CX_OFFSET =
        0;

    constexpr uint64_t CY_OFFSET =
        NUM_PREDICTIONS;

    constexpr uint64_t WIDTH_OFFSET =
        2ULL * NUM_PREDICTIONS;

    constexpr uint64_t HEIGHT_OFFSET =
        3ULL * NUM_PREDICTIONS;

    const float imageMaxX =
        static_cast<float>(
            preprocessInfo_.originalWidth - 1
        );

    const float imageMaxY =
        static_cast<float>(
            preprocessInfo_.originalHeight - 1
        );

    const float scale =
        preprocessInfo_.scale;

    const float padLeft =
        static_cast<float>(
            preprocessInfo_.padLeft
        );

    const float padTop =
        static_cast<float>(
            preprocessInfo_.padTop
        );

    proposals.reserve(
        NUM_PREDICTIONS
    );

    for (uint64_t predictionIndex = 0;
         predictionIndex < NUM_PREDICTIONS;
         ++predictionIndex) {

        float score = 0.0F;

        if (!readTensorValue(
                *conf,
                predictionIndex,
                score
            )) {

            lastError_ =
                "Failed to read conf_out value";

            return false;
        }

        if (!std::isfinite(
                score
            )) {

            continue;
        }

        if (score <
            confidenceThreshold) {

            continue;
        }

        float cx = 0.0F;
        float cy = 0.0F;
        float width = 0.0F;
        float height = 0.0F;

        if (!readTensorValue(
                *boxes,
                CX_OFFSET +
                    predictionIndex,
                cx
            ) ||
            !readTensorValue(
                *boxes,
                CY_OFFSET +
                    predictionIndex,
                cy
            ) ||
            !readTensorValue(
                *boxes,
                WIDTH_OFFSET +
                    predictionIndex,
                width
            ) ||
            !readTensorValue(
                *boxes,
                HEIGHT_OFFSET +
                    predictionIndex,
                height
            )) {

            lastError_ =
                "Failed to read boxes_out value";

            return false;
        }

        if (!std::isfinite(cx) ||
            !std::isfinite(cy) ||
            !std::isfinite(width) ||
            !std::isfinite(height)) {

            continue;
        }

        float x1 =
            cx
            -
            width / 2.0F;

        float y1 =
            cy
            -
            height / 2.0F;

        float x2 =
            cx
            +
            width / 2.0F;

        float y2 =
            cy
            +
            height / 2.0F;

        // Undo centered letterbox.

        x1 =
            (
                x1
                -
                padLeft
            )
            /
            scale;

        x2 =
            (
                x2
                -
                padLeft
            )
            /
            scale;

        y1 =
            (
                y1
                -
                padTop
            )
            /
            scale;

        y2 =
            (
                y2
                -
                padTop
            )
            /
            scale;

        x1 =
            std::clamp(
                x1,
                0.0F,
                imageMaxX
            );

        x2 =
            std::clamp(
                x2,
                0.0F,
                imageMaxX
            );

        y1 =
            std::clamp(
                y1,
                0.0F,
                imageMaxY
            );

        y2 =
            std::clamp(
                y2,
                0.0F,
                imageMaxY
            );

        // Remove degenerate boxes.

        if (!(x2 > x1) ||
            !(y2 > y1)) {

            continue;
        }

        PersonDetectionProposal proposal;

        proposal.score =
            score;

        proposal.bbox = {
            x1,
            y1,
            x2,
            y2
        };

        proposals.push_back(
            proposal
        );
    }

    lastError_.clear();

    return true;
}

bool PersonDetectionModel::applyNms(
    const std::vector<PersonDetectionProposal>& proposals,
    std::vector<PersonDetectionProposal>& kept,
    float nmsThreshold
)
{
    kept.clear();

    if (!std::isfinite(
            nmsThreshold
        ) ||
        nmsThreshold < 0.0F ||
        nmsThreshold > 1.0F) {

        lastError_ =
            "NMS threshold must be in [0, 1]";

        return false;
    }

    if (proposals.empty()) {

        lastError_.clear();

        return true;
    }

    // Python:
    //
    // order = scores.argsort()[::-1]
    //
    // Do not depend on original YOLO prediction order.

    std::vector<std::size_t> order(
        proposals.size()
    );

    std::iota(
        order.begin(),
        order.end(),
        0
    );

    std::sort(
        order.begin(),
        order.end(),
        [&proposals](
            std::size_t lhs,
            std::size_t rhs
        ) {
            return
                proposals[lhs].score
                >
                proposals[rhs].score;
        }
    );

    std::vector<uint8_t> suppressed(
        proposals.size(),
        0
    );

    kept.reserve(
        proposals.size()
    );

    for (std::size_t orderPosition = 0;
         orderPosition < order.size();
         ++orderPosition) {

        const std::size_t currentIndex =
            order[
                orderPosition
            ];

        if (suppressed[
                currentIndex
            ] != 0) {

            continue;
        }

        kept.push_back(
            proposals[
                currentIndex
            ]
        );

        for (std::size_t nextPosition =
                 orderPosition + 1;
             nextPosition < order.size();
             ++nextPosition) {

            const std::size_t candidateIndex =
                order[
                    nextPosition
                ];

            if (suppressed[
                    candidateIndex
                ] != 0) {

                continue;
            }

            const float iou =
                calculateIoU(
                    proposals[
                        currentIndex
                    ].bbox,

                    proposals[
                        candidateIndex
                    ].bbox
                );

            // Python:
            //
            // order = order[1:][
            //     iou <= iou_thresh
            // ]
            //
            // Therefore suppress when:
            //
            // iou > threshold

            if (iou >
                nmsThreshold) {

                suppressed[
                    candidateIndex
                ] = 1;
            }
        }
    }

    lastError_.clear();

    return true;
}

float PersonDetectionModel::calculateIoU(
    const std::array<float, 4>& lhs,
    const std::array<float, 4>& rhs
) noexcept
{
    // IMPORTANT:
    //
    // Person YOLO Python NMS does NOT use +1.
    //
    // area =
    //   max(0, x2-x1)
    //   *
    //   max(0, y2-y1)

    const float lhsWidth =
        std::max(
            0.0F,
            lhs[2]
                -
                lhs[0]
        );

    const float lhsHeight =
        std::max(
            0.0F,
            lhs[3]
                -
                lhs[1]
        );

    const float rhsWidth =
        std::max(
            0.0F,
            rhs[2]
                -
                rhs[0]
        );

    const float rhsHeight =
        std::max(
            0.0F,
            rhs[3]
                -
                rhs[1]
        );

    const float lhsArea =
        lhsWidth
        *
        lhsHeight;

    const float rhsArea =
        rhsWidth
        *
        rhsHeight;

    const float xx1 =
        std::max(
            lhs[0],
            rhs[0]
        );

    const float yy1 =
        std::max(
            lhs[1],
            rhs[1]
        );

    const float xx2 =
        std::min(
            lhs[2],
            rhs[2]
        );

    const float yy2 =
        std::min(
            lhs[3],
            rhs[3]
        );

    const float intersectionWidth =
        std::max(
            0.0F,
            xx2
                -
                xx1
        );

    const float intersectionHeight =
        std::max(
            0.0F,
            yy2
                -
                yy1
        );

    const float intersection =
        intersectionWidth
        *
        intersectionHeight;

    const float unionArea =
        lhsArea
        +
        rhsArea
        -
        intersection;

    // Python:
    //
    // iou =
    //   inter /
    //   max(union, 1e-9)

    return
        intersection
        /
        std::max(
            unionArea,
            1e-9F
        );
}

bool PersonDetectionModel::postprocess(
    std::vector<PersonDetectionResult>& results,
    float confidenceThreshold,
    float nmsThreshold
)
{
    results.clear();

    std::vector<
        PersonDetectionProposal
    > proposals;

    if (!decode(
            proposals,
            confidenceThreshold
        )) {

        return false;
    }

    std::vector<
        PersonDetectionProposal
    > kept;

    if (!applyNms(
            proposals,
            kept,
            nmsThreshold
        )) {

        return false;
    }

    results.reserve(
        kept.size()
    );

    for (const auto& proposal :
         kept) {

        PersonDetectionResult result;

        result.score =
            proposal.score;

        result.bbox =
            proposal.bbox;

        results.push_back(
            result
        );
    }

    lastError_.clear();

    return true;
}

bool PersonDetectionModel::detect(
    const cv::Mat& bgrImage,
    std::vector<PersonDetectionResult>& results,
    float confidenceThreshold,
    float nmsThreshold
)
{
    results.clear();

    if (!preprocess(
            bgrImage
        )) {

        return false;
    }

    if (!infer()) {

        return false;
    }

    if (!postprocess(
            results,
            confidenceThreshold,
            nmsThreshold
        )) {

        return false;
    }

    lastError_.clear();

    return true;
}

cv::Mat PersonDetectionModel::renderDetections(
    const cv::Mat& bgrImage,
    const std::vector<PersonDetectionResult>& results
) const
{
    if (bgrImage.empty()) {
        return {};
    }

    cv::Mat rendered =
        bgrImage.clone();

    for (const auto& result :
         results) {

        // Python:
        //
        // p1 = (
        //   int(round(x1)),
        //   int(round(y1))
        // )

        const int x1 =
            pythonRound(
                result.bbox[0]
            );

        const int y1 =
            pythonRound(
                result.bbox[1]
            );

        const int x2 =
            pythonRound(
                result.bbox[2]
            );

        const int y2 =
            pythonRound(
                result.bbox[3]
            );

        cv::rectangle(
            rendered,
            cv::Point(
                x1,
                y1
            ),
            cv::Point(
                x2,
                y2
            ),
            cv::Scalar(
                0,
                255,
                0
            ),
            2
        );

        std::ostringstream label;

        label
            << "person "
            << std::fixed
            << std::setprecision(2)
            << result.score;

        cv::putText(
            rendered,
            label.str(),
            cv::Point(
                x1,
                std::max(
                    15,
                    y1 - 5
                )
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(
                0,
                255,
                0
            ),
            1,
            cv::LINE_AA
        );
    }

    return rendered;
}

bool PersonDetectionModel::readTensorValue(
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

void PersonDetectionModel::shutdown()
{
    executionInputTensors_.clear();

    executionOutputTensors_.clear();

    inputBuffers_.clear();

    outputBuffers_.clear();

    preprocessInfo_ = {};

    model_.shutdown();
}

bool PersonDetectionModel::ready() const noexcept
{
    return
        model_.graphsFinalized()
        &&
        inputBuffers_.size() == 1
        &&
        outputBuffers_.size() == 2
        &&
        executionInputTensors_.size() == 1
        &&
        executionOutputTensors_.size() == 2;
}

const PersonDetectionPreprocessInfo&
PersonDetectionModel::preprocessInfo() const noexcept
{
    return
        preprocessInfo_;
}

const inference::QnnTensorBuffer*
PersonDetectionModel::inputBuffer() const noexcept
{
    if (inputBuffers_.empty()) {
        return nullptr;
    }

    return
        inputBuffers_[0].get();
}

std::size_t
PersonDetectionModel::outputCount() const noexcept
{
    return
        outputBuffers_.size();
}

const inference::QnnTensorBuffer*
PersonDetectionModel::outputBuffer(
    std::size_t index
) const noexcept
{
    if (index >=
        outputBuffers_.size()) {

        return nullptr;
    }

    return
        outputBuffers_[
            index
        ].get();
}

const inference::QnnTensorBuffer*
PersonDetectionModel::outputBufferByName(
    const std::string& name
) const noexcept
{
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

const std::string&
PersonDetectionModel::lastError() const noexcept
{
    return
        lastError_;
}

Qnn_DataType_t
PersonDetectionModel::dataType(
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

uint32_t PersonDetectionModel::rank(
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
        return 0;
    }
}

const uint32_t*
PersonDetectionModel::dimensions(
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
        return nullptr;
    }
}

const Qnn_QuantizeParams_t*
PersonDetectionModel::quantization(
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

int PersonDetectionModel::pythonRound(
    double value
) noexcept
{
    const double lower =
        std::floor(
            value
        );

    const double fraction =
        value
        -
        lower;

    if (fraction < 0.5) {

        return
            static_cast<int>(
                lower
            );
    }

    if (fraction > 0.5) {

        return
            static_cast<int>(
                lower + 1.0
            );
    }

    const auto lowerInteger =
        static_cast<int64_t>(
            lower
        );

    if (lowerInteger % 2 == 0) {

        return
            static_cast<int>(
                lowerInteger
            );
    }

    return
        static_cast<int>(
            lowerInteger + 1
        );
}

} // namespace models