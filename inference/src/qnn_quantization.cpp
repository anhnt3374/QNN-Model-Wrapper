#include "inference/qnn_quantization.hpp"

#include <cstdint>

namespace inference {

float dequantizeScaleOffset(
    uint16_t quantizedValue,
    float scale,
    int32_t offset
) noexcept
{
    const int64_t shiftedValue =
        static_cast<int64_t>(
            quantizedValue
        )
        +
        static_cast<int64_t>(
            offset
        );

    return
        static_cast<float>(
            shiftedValue
        )
        *
        scale;
}

} // namespace inference