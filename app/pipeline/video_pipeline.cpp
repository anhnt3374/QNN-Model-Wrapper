#include "pipeline/video_pipeline.hpp"

#include "metrics/metrics.hpp"
#include "video/frame.hpp"

#include <vector>

namespace pipeline {

VideoPipeline::VideoPipeline(
    video::FrameSource& source,
    video::FrameSink& sink,
    metrics::PipelineProfiler* profiler,
    metrics::MetricsLogger* metricsLogger,
    FrameProcessor* processor
) noexcept
    : source_(
          source
      ),
      sink_(
          sink
      ),
      profiler_(
          profiler
      ),
      metricsLogger_(
          metricsLogger
      ),
      processor_(
          processor
      )
{
}

bool VideoPipeline::run()
{
    stats_ = {};

    lastError_.clear();

    if (metricsLogger_ != nullptr &&
        !metricsLogger_->isOpen()) {

        lastError_ =
            "MetricsLogger was provided but is not open";

        return false;
    }

    if (profiler_ != nullptr) {

        profiler_->startRun();
    }

    // =====================================================
    // Source
    // =====================================================

    if (!source_.open()) {

        lastError_ =
            "Cannot open frame source: "
            +
            source_.lastError();

        if (profiler_ != nullptr) {
            profiler_->finishRun();
        }

        return false;
    }

    // =====================================================
    // Processor
    // =====================================================

    if (processor_ != nullptr) {

        if (!processor_->configure(
                source_.fps(),
                source_.width(),
                source_.height()
            )) {

            lastError_ =
                "Cannot configure frame processor: "
                +
                processor_->lastError();

            source_.close();

            if (profiler_ != nullptr) {
                profiler_->finishRun();
            }

            return false;
        }
    }

    // =====================================================
    // Sink
    // =====================================================

    if (!sink_.open(
            source_.fps(),
            source_.width(),
            source_.height()
        )) {

        lastError_ =
            "Cannot open frame sink: "
            +
            sink_.lastError();

        source_.close();

        if (profiler_ != nullptr) {
            profiler_->finishRun();
        }

        return false;
    }

    // =====================================================
    // Main loop
    // =====================================================

    while (true) {

        const Clock::time_point frameStart =
            Clock::now();

        // =================================================
        // READ ORIGINAL FRAME
        // =================================================

        const Clock::time_point readStart =
            Clock::now();

        video::Frame sourceFrame;

        const bool readSuccess =
            source_.read(
                sourceFrame
            );

        const Clock::time_point readEnd =
            Clock::now();

        if (!readSuccess) {

            if (!source_.lastError().empty()) {

                lastError_ =
                    "Frame source read failed: "
                    +
                    source_.lastError();

                sink_.close();
                source_.close();

                if (profiler_ != nullptr) {
                    profiler_->finishRun();
                }

                return false;
            }

            break;
        }

        ++stats_.framesRead;

        const double readMs =
            durationMs(
                readStart,
                readEnd
            );

        // =================================================
        // RENDER FRAME
        //
        // Initially references the same original image.
        //
        // Model-specific renderDetections() returns a new
        // rendered Mat when it actually draws something.
        //
        // AI models always receive sourceFrame, so later
        // overlays can never contaminate another model's
        // input.
        // =================================================

        video::Frame renderFrame =
            sourceFrame;

        // =================================================
        // PROCESS
        // =================================================

        const Clock::time_point processStart =
            Clock::now();

        std::vector<
            metrics::ModelMetrics
        > modelMetrics;

        if (processor_ != nullptr) {

            if (!processor_->process(
                    sourceFrame,
                    renderFrame,
                    modelMetrics
                )) {

                lastError_ =
                    "Frame processor failed: "
                    +
                    processor_->lastError();

                sink_.close();
                source_.close();

                if (profiler_ != nullptr) {
                    profiler_->finishRun();
                }

                return false;
            }
        }

        const Clock::time_point processEnd =
            Clock::now();

        const double processMs =
            durationMs(
                processStart,
                processEnd
            );

        // =================================================
        // MODEL CSV
        // =================================================

        if (metricsLogger_ != nullptr) {

            for (const auto& metric :
                 modelMetrics) {

                if (!metricsLogger_->writeModel(
                        metric
                    )) {

                    lastError_ =
                        "Cannot write model metrics: "
                        +
                        metricsLogger_->lastError();

                    sink_.close();
                    source_.close();

                    if (profiler_ != nullptr) {
                        profiler_->finishRun();
                    }

                    return false;
                }
            }
        }

        // =================================================
        // WRITE RENDER FRAME
        // =================================================

        const Clock::time_point writeStart =
            Clock::now();

        if (!sink_.write(
                renderFrame
            )) {

            lastError_ =
                "Frame sink write failed: "
                +
                sink_.lastError();

            sink_.close();
            source_.close();

            if (profiler_ != nullptr) {
                profiler_->finishRun();
            }

            return false;
        }

        const Clock::time_point writeEnd =
            Clock::now();

        ++stats_.framesWritten;

        const double writeMs =
            durationMs(
                writeStart,
                writeEnd
            );

        const double totalMs =
            durationMs(
                frameStart,
                writeEnd
            );

        // =================================================
        // FRAME METRICS
        // =================================================

        metrics::FrameMetrics frameMetrics;

        frameMetrics.frameId =
            sourceFrame.frameId;

        frameMetrics.captureTimestampUs =
            sourceFrame.captureTimestampUs;

        frameMetrics.readMs =
            readMs;

        frameMetrics.processMs =
            processMs;

        frameMetrics.writeMs =
            writeMs;

        frameMetrics.totalMs =
            totalMs;

        if (totalMs > 0.0) {

            frameMetrics.instantaneousFps =
                1000.0
                /
                totalMs;
        }

        if (profiler_ != nullptr) {

            profiler_->recordFrame(
                frameMetrics
            );
        }

        if (metricsLogger_ != nullptr) {

            if (!metricsLogger_->writeFrame(
                    frameMetrics
                )) {

                lastError_ =
                    "Cannot write frame metrics: "
                    +
                    metricsLogger_->lastError();

                sink_.close();
                source_.close();

                if (profiler_ != nullptr) {
                    profiler_->finishRun();
                }

                return false;
            }
        }

        // =================================================
        // SYSTEM METRICS
        // =================================================

        if (profiler_ != nullptr &&
            metricsLogger_ != nullptr) {

            metrics::SystemMetrics systemMetrics;

            if (profiler_->sampleSystemIfDue(
                    systemMetrics
                )) {

                if (!metricsLogger_->writeSystem(
                        systemMetrics
                    )) {

                    lastError_ =
                        "Cannot write system metrics: "
                        +
                        metricsLogger_->lastError();

                    sink_.close();
                    source_.close();
                    profiler_->finishRun();

                    return false;
                }
            }
        }
    }

    sink_.close();

    source_.close();

    if (profiler_ != nullptr) {

        if (metricsLogger_ != nullptr) {

            metrics::SystemMetrics finalSystemMetrics;

            if (profiler_->sampleSystemIfDue(
                    finalSystemMetrics,
                    true
                )) {

                if (!metricsLogger_->writeSystem(
                        finalSystemMetrics
                    )) {

                    lastError_ =
                        "Cannot write final system metrics: "
                        +
                        metricsLogger_->lastError();

                    profiler_->finishRun();

                    return false;
                }
            }
        }

        profiler_->finishRun();
    }

    if (metricsLogger_ != nullptr) {

        metricsLogger_->flush();
    }

    if (stats_.framesRead == 0) {

        lastError_ =
            "Input video produced zero frames";

        return false;
    }

    if (stats_.framesWritten !=
        stats_.framesRead) {

        lastError_ =
            "Output frame count does not match input";

        return false;
    }

    return true;
}

const VideoPipelineStats&
VideoPipeline::stats() const noexcept
{
    return stats_;
}

const std::string&
VideoPipeline::lastError() const noexcept
{
    return lastError_;
}

double VideoPipeline::durationMs(
    Clock::time_point start,
    Clock::time_point end
) noexcept
{
    return
        std::chrono::duration<
            double,
            std::milli
        >(
            end - start
        ).count();
}

} // namespace pipeline