#include "pipeline/video_pipeline.hpp"

#include "metrics/metrics.hpp"
#include "video/frame.hpp"

namespace pipeline {

VideoPipeline::VideoPipeline(
    video::FrameSource& source,
    video::FrameSink& sink,
    metrics::PipelineProfiler* profiler,
    metrics::MetricsLogger* metricsLogger
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
      )
{
}

bool VideoPipeline::run()
{
    stats_ = {};

    lastError_.clear();

    // =====================================================
    // Metrics must already be opened by application.
    // =====================================================

    if (metricsLogger_ != nullptr &&
        !metricsLogger_->isOpen()) {

        lastError_ =
            "MetricsLogger was provided but is not open";

        return false;
    }

    // =====================================================
    // Start profiler
    // =====================================================

    if (profiler_ != nullptr) {

        profiler_->startRun();
    }

    // =====================================================
    // Open source
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
    // Open sink
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
        // READ
        // =================================================

        const Clock::time_point readStart =
            Clock::now();

        video::Frame frame;

        const bool readSuccess =
            source_.read(
                frame
            );

        const Clock::time_point readEnd =
            Clock::now();

        if (!readSuccess) {

            // Empty error means normal EOF.
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
        // PROCESS
        //
        // V2 intentionally has no AI.
        //
        // This timing slot becomes:
        //
        // scheduler
        // people
        // fire/smoke
        // attendance
        // renderer
        //
        // in later checkpoints.
        // =================================================

        const Clock::time_point processStart =
            Clock::now();

        // No processing in V2.

        const Clock::time_point processEnd =
            Clock::now();

        const double processMs =
            durationMs(
                processStart,
                processEnd
            );

        // =================================================
        // WRITE
        // =================================================

        const Clock::time_point writeStart =
            Clock::now();

        if (!sink_.write(
                frame
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

        const Clock::time_point frameEnd =
            writeEnd;

        const double totalMs =
            durationMs(
                frameStart,
                frameEnd
            );

        // =================================================
        // Frame metrics
        // =================================================

        metrics::FrameMetrics frameMetrics;

        frameMetrics.frameId =
            frame.frameId;

        frameMetrics.captureTimestampUs =
            frame.captureTimestampUs;

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

        // =================================================
        // Profiler
        // =================================================

        if (profiler_ != nullptr) {

            profiler_->recordFrame(
                frameMetrics
            );
        }

        // =================================================
        // CSV
        // =================================================

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
        // System sampling
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

    // =====================================================
    // Finish
    // =====================================================

    sink_.close();

    source_.close();

    if (profiler_ != nullptr) {

        // Force one final system sample so short videos
        // still contain resource information.
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

    lastError_.clear();

    return true;
}

const VideoPipelineStats&
VideoPipeline::stats() const noexcept
{
    return
        stats_;
}

const std::string&
VideoPipeline::lastError() const noexcept
{
    return
        lastError_;
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