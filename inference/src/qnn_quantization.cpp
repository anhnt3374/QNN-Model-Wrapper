#include "inference/qnn_quantization.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

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

uint16_t quantizeScaleOffsetToU16(
    float realValue,
    float scale,
    int32_t offset
) noexcept
{
    if (!std::isfinite(realValue) ||
        !std::isfinite(scale) ||
        scale <= 0.0F) {

        return 0;
    }

    const double quantized =
        std::round(
            static_cast<double>(
                realValue
            )
            /
            static_cast<double>(
                scale
            )
            -
            static_cast<double>(
                offset
            )
        );

    const double clamped =
        std::clamp(
            quantized,
            0.0,
            static_cast<double>(
                std::numeric_limits<uint16_t>::max()
            )
        );

    return
        static_cast<uint16_t>(
            clamped
        );
}

} // namespace inference