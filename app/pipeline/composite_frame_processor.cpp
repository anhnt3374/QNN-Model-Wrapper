#include "pipeline/composite_frame_processor.hpp"

namespace pipeline {

void CompositeFrameProcessor::addProcessor(
    FrameProcessor& processor
)
{
    processors_.push_back(
        &processor
    );
}

bool CompositeFrameProcessor::configure(
    double sourceFps,
    int width,
    int height
)
{
    lastError_.clear();

    for (std::size_t i = 0;
         i < processors_.size();
         ++i) {

        FrameProcessor* processor =
            processors_[i];

        if (processor == nullptr) {

            lastError_ =
                "CompositeFrameProcessor contains null processor";

            return false;
        }

        if (!processor->configure(
                sourceFps,
                width,
                height
            )) {

            lastError_ =
                "Processor["
                +
                std::to_string(i)
                +
                "] configure failed: "
                +
                processor->lastError();

            return false;
        }
    }

    return true;
}

bool CompositeFrameProcessor::process(
    const video::Frame& sourceFrame,
    video::Frame& renderFrame,
    std::vector<metrics::ModelMetrics>& modelMetrics
)
{
    lastError_.clear();

    for (std::size_t i = 0;
         i < processors_.size();
         ++i) {

        FrameProcessor* processor =
            processors_[i];

        if (processor == nullptr) {

            lastError_ =
                "CompositeFrameProcessor contains null processor";

            return false;
        }

        if (!processor->process(
                sourceFrame,
                renderFrame,
                modelMetrics
            )) {

            lastError_ =
                "Processor["
                +
                std::to_string(i)
                +
                "] process failed: "
                +
                processor->lastError();

            return false;
        }
    }

    return true;
}

const std::string&
CompositeFrameProcessor::lastError() const noexcept
{
    return lastError_;
}

std::size_t
CompositeFrameProcessor::processorCount() const noexcept
{
    return processors_.size();
}

} // namespace pipeline