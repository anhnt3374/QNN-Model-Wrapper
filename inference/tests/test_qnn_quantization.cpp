#include "inference/qnn_quantization.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>

namespace {

bool nearlyEqual(
    float actual,
    float expected,
    float tolerance = 1e-6F
)
{
    return
        std::fabs(
            actual - expected
        )
        <= tolerance;
}

bool expectFloat(
    const char* name,
    float actual,
    float expected,
    float tolerance = 1e-6F
)
{
    if (!nearlyEqual(
            actual,
            expected,
            tolerance
        )) {

        std::cerr
            << "[ERROR] "
            << name
            << '\n'
            << "        expected: "
            << expected
            << '\n'
            << "        actual: "
            << actual
            << '\n';

        return false;
    }

    std::cout
        << "[PASS] "
        << name
        << '\n';

    return true;
}

bool expectU16(
    const char* name,
    uint16_t actual,
    uint16_t expected
)
{
    if (actual != expected) {

        std::cerr
            << "[ERROR] "
            << name
            << '\n'
            << "        expected: "
            << expected
            << '\n'
            << "        actual: "
            << actual
            << '\n';

        return false;
    }

    std::cout
        << "[PASS] "
        << name
        << '\n';

    return true;
}

} // namespace

int main()
{
    // =====================================================
    // Dequantization
    // =====================================================

    const float realZero =
        inference::dequantizeScaleOffset(
            32768,
            3.03988e-05F,
            -32768
        );

    if (!expectFloat(
            "quantized zero becomes real zero",
            realZero,
            0.0F
        )) {

        return 1;
    }

    const float positiveValue =
        inference::dequantizeScaleOffset(
            1200,
            0.01F,
            -200
        );

    if (!expectFloat(
            "positive scale-offset value",
            positiveValue,
            10.0F
        )) {

        return 1;
    }

    const float negativeValue =
        inference::dequantizeScaleOffset(
            0,
            0.1F,
            -10
        );

    if (!expectFloat(
            "negative dequantized value",
            negativeValue,
            -1.0F
        )) {

        return 1;
    }

    // =====================================================
    // Quantization
    // =====================================================

    const uint16_t quantizedZero =
        inference::quantizeScaleOffsetToU16(
            0.0F,
            3.03988e-05F,
            -32768
        );

    if (!expectU16(
            "real zero becomes quantized zero",
            quantizedZero,
            32768
        )) {

        return 1;
    }

    const uint16_t quantizedPositive =
        inference::quantizeScaleOffsetToU16(
            10.0F,
            0.01F,
            -200
        );

    if (!expectU16(
            "positive real value quantizes correctly",
            quantizedPositive,
            1200
        )) {

        return 1;
    }

    // SCRFD normalization:
    //
    // black:
    // (0 - 127.5) / 128
    //
    // should map very close to the bottom of uint16.
    const float blackNormalized =
        (0.0F - 127.5F)
        /
        128.0F;

    const uint16_t blackQuantized =
        inference::quantizeScaleOffsetToU16(
            blackNormalized,
            3.03988e-05F,
            -32768
        );

    if (!expectU16(
            "SCRFD black pixel clamps to zero",
            blackQuantized,
            0
        )) {

        return 1;
    }

    // White is very close to / slightly beyond the
    // representable upper edge because of the actual
    // scale emitted by QNN, so it must clamp to 65535.
    const float whiteNormalized =
        (255.0F - 127.5F)
        /
        128.0F;

    const uint16_t whiteQuantized =
        inference::quantizeScaleOffsetToU16(
            whiteNormalized,
            3.03988e-05F,
            -32768
        );

    if (!expectU16(
            "SCRFD white pixel clamps to uint16 max",
            whiteQuantized,
            65535
        )) {

        return 1;
    }

    std::cout
        << "[PASS] QNN quantization tests complete\n";

    return 0;
}