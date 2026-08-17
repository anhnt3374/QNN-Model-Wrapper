#pragma once

#include "metrics/metrics.hpp"
#include "models/fire_smoke_detection_model.hpp"
#include "pipeline/frame_processor.hpp"
#include "async_runtime/htp_execution_gate.hpp"

#include <chrono>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace pipeline {

class FireSmokePipeline final
    : public FrameProcessor {
public:
    explicit FireSmokePipeline(
        models::FireSmokeDetectionModel& model,
        double targetInferenceFps = 5.0,
        std::size_t temporalWindow = 5,
        std::size_t confirmationCount = 3
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

    bool fireConfirmed() const noexcept;

    bool smokeConfirmed() const noexcept;

    std::size_t firePositiveCount() const noexcept;

    std::size_t smokePositiveCount() const noexcept;

    void setHtpExecutionGate(
        async_runtime::HtpExecutionGate* gate
    ) noexcept;

private:
    using Clock =
        std::chrono::steady_clock;

    bool shouldRunInference(
        const video::Frame& frame
    );

    void updateTemporalState(
        const std::vector<
            models::FireSmokeDetectionResult
        >& detections
    );

    void renderStatusOverlay(
        cv::Mat& image
    ) const;

    static std::size_t countTrue(
        const std::deque<bool>& history
    ) noexcept;

    static double durationMs(
        Clock::time_point start,
        Clock::time_point end
    ) noexcept;

    async_runtime::HtpExecutionGate* htpGate_ = nullptr;

private:
    models::FireSmokeDetectionModel& model_;

    double targetInferenceFps_ =
        5.0;

    int64_t inferencePeriodUs_ =
        200000;

    bool scheduleInitialized_ =
        false;

    int64_t nextInferenceTimestampUs_ =
        0;

    std::size_t temporalWindow_ =
        5;

    std::size_t confirmationCount_ =
        3;

    std::deque<bool> fireHistory_;

    std::deque<bool> smokeHistory_;

    bool fireConfirmed_ =
        false;

    bool smokeConfirmed_ =
        false;

    std::vector<
        models::FireSmokeDetectionResult
    > lastDetections_;

    uint64_t inferenceCount_ =
        0;

    std::string lastError_;
};

} // namespace pipeline