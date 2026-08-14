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

bool expectValue(
    const char* testName,
    float actual,
    float expected
)
{
    if (!nearlyEqual(
            actual,
            expected
        )) {

        std::cerr
            << "[ERROR] "
            << testName
            << '\n';

        std::cerr
            << "        expected: "
            << expected
            << '\n';

        std::cerr
            << "        actual: "
            << actual
            << '\n';

        return false;
    }

    std::cout
        << "[PASS] "
        << testName
        << '\n';

    return true;
}

} // namespace

int main()
{
    // =====================================================
    // Case 1
    //
    // SCRFD input zero point:
    //
    // q      = 32768
    // offset = -32768
    //
    // real = (32768 - 32768) * scale
    //      = 0
    // =====================================================

    const float realZero =
        inference::dequantizeScaleOffset(
            32768,
            3.03988e-05F,
            -32768
        );

    if (!expectValue(
            "quantized zero becomes real zero",
            realZero,
            0.0F
        )) {

        return 1;
    }

    // =====================================================
    // Case 2
    //
    // (1200 - 200) * 0.01
    // = 10
    // =====================================================

    const float positiveValue =
        inference::dequantizeScaleOffset(
            1200,
            0.01F,
            -200
        );

    if (!expectValue(
            "positive scale-offset value",
            positiveValue,
            10.0F
        )) {

        return 1;
    }

    // =====================================================
    // Case 3
    //
    // (0 - 10) * 0.1
    // = -1
    // =====================================================

    const float negativeValue =
        inference::dequantizeScaleOffset(
            0,
            0.1F,
            -10
        );

    if (!expectValue(
            "negative dequantized value",
            negativeValue,
            -1.0F
        )) {

        return 1;
    }

    // =====================================================
    // Case 4
    //
    // Real SCRFD score_8 sample from 5G.1:
    //
    // q      = 704
    // scale  = 1.52588e-05
    // offset = 0
    //
    // expected ≈ 0.0107421952
    // =====================================================

    const float scrfdScore =
        inference::dequantizeScaleOffset(
            704,
            1.52588e-05F,
            0
        );

    if (!expectValue(
            "SCRFD score sample",
            scrfdScore,
            0.0107421952F
        )) {

        return 1;
    }

    std::cout
        << "[PASS] QNN scale-offset "
        << "dequantization tests complete\n";

    return 0;
}