#pragma once

#include <QnnTypes.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace inference {

class QnnTensorBuffer {
public:
    QnnTensorBuffer() = default;

    ~QnnTensorBuffer() = default;

    QnnTensorBuffer(
        const QnnTensorBuffer&
    ) = delete;

    QnnTensorBuffer& operator=(
        const QnnTensorBuffer&
    ) = delete;

    QnnTensorBuffer(
        QnnTensorBuffer&&
    ) = delete;

    QnnTensorBuffer& operator=(
        QnnTensorBuffer&&
    ) = delete;

    bool initialize(
        const Qnn_Tensor_t& source
    );

    void reset();

    bool ready() const noexcept;

    Qnn_Tensor_t& tensor() noexcept;

    const Qnn_Tensor_t&
    tensor() const noexcept;

    void* data() noexcept;

    const void* data() const noexcept;

    uint64_t elementCount() const noexcept;

    uint32_t byteSize() const noexcept;

    uint32_t bytesPerElement() const noexcept;

    const std::string&
    name() const noexcept;

    const std::string&
    lastError() const noexcept;

private:
    static uint32_t getBytesPerElement(
        Qnn_DataType_t dataType
    ) noexcept;

    static bool isSupportedQuantization(
        const Qnn_QuantizeParams_t& params
    ) noexcept;

    bool initializeV1(
        const Qnn_TensorV1_t& source
    );

    bool initializeV2(
        const Qnn_TensorV2_t& source
    );

    bool allocateBuffer(
        Qnn_DataType_t dataType,
        uint32_t rank,
        const uint32_t* dimensions
    );

private:
    Qnn_Tensor_t tensor_{};

    std::string name_;

    std::vector<uint32_t> dimensions_;

    std::vector<uint8_t> dynamicDimensions_;

    std::vector<uint8_t> buffer_;

    uint64_t elementCount_ = 0;

    uint32_t byteSize_ = 0;

    uint32_t bytesPerElement_ = 0;

    bool ready_ = false;

    std::string lastError_;
};

} // namespace inference