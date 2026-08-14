#include "inference/qnn_tensor_buffer.hpp"

#include <limits>
#include <sstream>

namespace inference {

bool QnnTensorBuffer::initialize(
    const Qnn_Tensor_t& source
)
{
    reset();

    switch (source.version) {

    case QNN_TENSOR_VERSION_1:

        if (!initializeV1(
                source.v1
            )) {

            return false;
        }

        break;

    case QNN_TENSOR_VERSION_2:

        if (!initializeV2(
                source.v2
            )) {

            return false;
        }

        break;

    default: {

        std::ostringstream oss;

        oss
            << "Unsupported QNN tensor version: "
            << static_cast<int>(
                   source.version
               );

        lastError_ = oss.str();

        return false;
    }
    }

    ready_ = true;

    lastError_.clear();

    return true;
}

bool QnnTensorBuffer::initializeV1(
    const Qnn_TensorV1_t& source
)
{
    if (!isSupportedQuantization(
            source.quantizeParams
        )) {

        lastError_ =
            "Unsupported quantization encoding";

        return false;
    }

    if (!allocateBuffer(
            source.dataType,
            source.rank,
            source.dimensions
        )) {

        return false;
    }

    tensor_.version =
        QNN_TENSOR_VERSION_1;

    tensor_.v1 =
        source;

    // -----------------------------------------------------
    // Own tensor name
    // -----------------------------------------------------

    if (source.name != nullptr) {
        name_ = source.name;
    }

    tensor_.v1.name =
        name_.empty()
            ? nullptr
            : name_.c_str();

    // -----------------------------------------------------
    // Own dimensions
    // -----------------------------------------------------

    if (source.rank > 0) {
        dimensions_.assign(
            source.dimensions,
            source.dimensions +
                source.rank
        );
    }

    tensor_.v1.dimensions =
        dimensions_.empty()
            ? nullptr
            : dimensions_.data();

    // -----------------------------------------------------
    // Runtime memory
    // -----------------------------------------------------

    tensor_.v1.memType =
        QNN_TENSORMEMTYPE_RAW;

    tensor_.v1.clientBuf.data =
        buffer_.data();

    tensor_.v1.clientBuf.dataSize =
        byteSize_;

    return true;
}

bool QnnTensorBuffer::initializeV2(
    const Qnn_TensorV2_t& source
)
{
    if (!isSupportedQuantization(
            source.quantizeParams
        )) {

        lastError_ =
            "Unsupported quantization encoding";

        return false;
    }

    if (!allocateBuffer(
            source.dataType,
            source.rank,
            source.dimensions
        )) {

        return false;
    }

    tensor_.version =
        QNN_TENSOR_VERSION_2;

    tensor_.v2 =
        source;

    // -----------------------------------------------------
    // Own tensor name
    // -----------------------------------------------------

    if (source.name != nullptr) {
        name_ = source.name;
    }

    tensor_.v2.name =
        name_.empty()
            ? nullptr
            : name_.c_str();

    // -----------------------------------------------------
    // Own dimensions
    // -----------------------------------------------------

    if (source.rank > 0) {
        dimensions_.assign(
            source.dimensions,
            source.dimensions +
                source.rank
        );
    }

    tensor_.v2.dimensions =
        dimensions_.empty()
            ? nullptr
            : dimensions_.data();

    // -----------------------------------------------------
    // Own dynamic dimension flags
    // -----------------------------------------------------

    if (source.isDynamicDimensions != nullptr &&
        source.rank > 0) {

        dynamicDimensions_.assign(
            source.isDynamicDimensions,
            source.isDynamicDimensions +
                source.rank
        );
    }

    tensor_.v2.isDynamicDimensions =
        dynamicDimensions_.empty()
            ? nullptr
            : dynamicDimensions_.data();

    // -----------------------------------------------------
    // Runtime memory
    // -----------------------------------------------------

    tensor_.v2.memType =
        QNN_TENSORMEMTYPE_RAW;

    tensor_.v2.clientBuf.data =
        buffer_.data();

    tensor_.v2.clientBuf.dataSize =
        byteSize_;

    // Before graph execution this is not meaningful.
    tensor_.v2.isProduced = 0;

    return true;
}

bool QnnTensorBuffer::allocateBuffer(
    Qnn_DataType_t dataType,
    uint32_t rank,
    const uint32_t* dimensions
)
{
    bytesPerElement_ =
        getBytesPerElement(
            dataType
        );

    if (bytesPerElement_ == 0) {
        std::ostringstream oss;

        oss
            << "Unsupported QNN datatype: "
            << static_cast<int>(
                   dataType
               );

        lastError_ = oss.str();

        return false;
    }

    if (rank > 0 &&
        dimensions == nullptr) {

        lastError_ =
            "Tensor dimensions are null";

        return false;
    }

    uint64_t count = 1;

    if (rank == 0) {
        count = 1;
    } else {

        for (uint32_t i = 0;
             i < rank;
             ++i) {

            if (dimensions[i] == 0) {
                lastError_ =
                    "Tensor contains zero dimension";

                return false;
            }

            if (count >
                std::numeric_limits<uint64_t>::max()
                    / dimensions[i]) {

                lastError_ =
                    "Tensor element count overflow";

                return false;
            }

            count *= dimensions[i];
        }
    }

    const uint64_t bytes =
        count *
        static_cast<uint64_t>(
            bytesPerElement_
        );

    if (bytes >
        std::numeric_limits<uint32_t>::max()) {

        lastError_ =
            "Tensor buffer is larger than "
            "Qnn_ClientBuffer_t::dataSize";

        return false;
    }

    elementCount_ =
        count;

    byteSize_ =
        static_cast<uint32_t>(
            bytes
        );

    try {
        buffer_.resize(
            byteSize_
        );
    } catch (...) {

        lastError_ =
            "Failed to allocate tensor buffer";

        reset();

        return false;
    }

    if (buffer_.empty()) {
        lastError_ =
            "Allocated tensor buffer is empty";

        return false;
    }

    return true;
}

uint32_t QnnTensorBuffer::getBytesPerElement(
    Qnn_DataType_t dataType
) noexcept
{
    switch (dataType) {

    // -----------------------------------------------------
    // 8 bit
    // -----------------------------------------------------

    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_UINT_8:

    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:

    case QNN_DATATYPE_BOOL_8:

        return 1;

    // -----------------------------------------------------
    // 16 bit
    // -----------------------------------------------------

    case QNN_DATATYPE_FLOAT_16:

    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_UINT_16:

    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:

        return 2;

    // -----------------------------------------------------
    // 32 bit
    // -----------------------------------------------------

    case QNN_DATATYPE_FLOAT_32:

    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_UINT_32:

    case QNN_DATATYPE_SFIXED_POINT_32:
    case QNN_DATATYPE_UFIXED_POINT_32:

        return 4;

    default:

        return 0;
    }
}

bool QnnTensorBuffer::isSupportedQuantization(
    const Qnn_QuantizeParams_t& params
) noexcept
{
    switch (params.quantizationEncoding) {

    case QNN_QUANTIZATION_ENCODING_SCALE_OFFSET:
        return true;

    case QNN_QUANTIZATION_ENCODING_UNDEFINED:
        return true;

    default:
        return false;
    }
}

void QnnTensorBuffer::reset()
{
    tensor_ = {};

    name_.clear();

    dimensions_.clear();

    dynamicDimensions_.clear();

    buffer_.clear();

    elementCount_ = 0;

    byteSize_ = 0;

    bytesPerElement_ = 0;

    ready_ = false;

    lastError_.clear();
}

bool QnnTensorBuffer::ready() const noexcept
{
    return ready_;
}

Qnn_Tensor_t&
QnnTensorBuffer::tensor() noexcept
{
    return tensor_;
}

const Qnn_Tensor_t&
QnnTensorBuffer::tensor() const noexcept
{
    return tensor_;
}

void*
QnnTensorBuffer::data() noexcept
{
    return
        buffer_.empty()
            ? nullptr
            : buffer_.data();
}

const void*
QnnTensorBuffer::data() const noexcept
{
    return
        buffer_.empty()
            ? nullptr
            : buffer_.data();
}

uint64_t
QnnTensorBuffer::elementCount() const noexcept
{
    return elementCount_;
}

uint32_t
QnnTensorBuffer::byteSize() const noexcept
{
    return byteSize_;
}

uint32_t
QnnTensorBuffer::bytesPerElement() const noexcept
{
    return bytesPerElement_;
}

const std::string&
QnnTensorBuffer::name() const noexcept
{
    return name_;
}

const std::string&
QnnTensorBuffer::lastError() const noexcept
{
    return lastError_;
}

} // namespace inference