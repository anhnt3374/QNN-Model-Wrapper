#pragma once

#include "metrics/metrics.hpp"
#include "video/frame.hpp"

#include <string>
#include <vector>

namespace pipeline {

class FrameProcessor {
public:
    virtual ~FrameProcessor() = default;

    FrameProcessor(
        const FrameProcessor&
    ) = delete;

    FrameProcessor& operator=(
        const FrameProcessor&
    ) = delete;

    virtual bool configure(
        double sourceFps,
        int width,
        int height
    ) = 0;

    // sourceFrame:
    //     immutable original frame.
    //     AI models MUST read from this frame.
    //
    // renderFrame:
    //     output frame.
    //     Processors may draw overlays on this frame.
    //
    // This separation is required once multiple models
    // process the same video frame.
    virtual bool process(
        const video::Frame& sourceFrame,
        video::Frame& renderFrame,
        std::vector<metrics::ModelMetrics>& modelMetrics
    ) = 0;

    virtual const std::string&
    lastError() const noexcept = 0;

protected:
    FrameProcessor() = default;
};

} // namespace pipeline