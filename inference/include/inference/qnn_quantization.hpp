#pragma once

#include <cstdint>

namespace inference {

float dequantizeScaleOffset(
    uint16_t quantizedValue,
    float scale,
    int32_t offset
) noexcept;

uint16_t quantizeScaleOffsetToU16(
    float realValue,
    float scale,
    int32_t offset
) noexcept;

} // namespace inference