#pragma once

#include "metrics/metrics.hpp"
#include "models/person_detection_model.hpp"
#include "pipeline/frame_processor.hpp"
#include "async_runtime/htp_execution_gate.hpp"

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace pipeline {

class PersonPipeline final
    : public FrameProcessor {
public:
    explicit PersonPipeline(
        models::PersonDetectionModel& model,
        double targetInferenceFps = 10.0
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

    uint64_t inferenceCount() const noexcept;

    uint64_t renderedFrameCount() const noexcept;

    double targetInferenceFps() const noexcept;
    
    void setHtpExecutionGate(
        async_runtime::HtpExecutionGate* gate
    ) noexcept;

private:
    using Clock =
        std::chrono::steady_clock;

    bool shouldRunInference(
        const video::Frame& frame
    );

    static double durationMs(
        Clock::time_point start,
        Clock::time_point end
    ) noexcept;

    async_runtime::HtpExecutionGate* htpGate_ = nullptr;

private:
    models::PersonDetectionModel& model_;

    double targetInferenceFps_ =
        10.0;

    int64_t inferencePeriodUs_ =
        100000;

    bool scheduleInitialized_ =
        false;

    int64_t nextInferenceTimestampUs_ =
        0;

    std::vector<
        models::PersonDetectionResult
    > lastDetections_;

    uint64_t inferenceCount_ =
        0;

    uint64_t renderedFrameCount_ =
        0;

    std::string lastError_;
};

} // namespace pipeline