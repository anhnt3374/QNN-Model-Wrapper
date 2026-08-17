#pragma once

#include "pipeline/frame_processor.hpp"

#include <string>
#include <vector>

namespace pipeline {

class CompositeFrameProcessor final
    : public FrameProcessor {
public:
    CompositeFrameProcessor() = default;

    void addProcessor(
        FrameProcessor& processor
    );

    bool configure(
        double sourceFps,
        int width,
        int height
    ) override;

    bool process(
        const video::Frame& sourceFrame,
        video::Frame& renderFrame,
        std::vector<metrics::ModelMetrics>& modelMetrics
    ) override;

    const std::string&
    lastError() const noexcept override;

    std::size_t processorCount() const noexcept;

private:
    std::vector<FrameProcessor*> processors_;

    std::string lastError_;
};

} // namespace pipeline