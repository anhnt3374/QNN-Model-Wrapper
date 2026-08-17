#include "async_runtime/async_metrics_logger.hpp"

#include <filesystem>
#include <iomanip>

namespace async_runtime {

AsyncMetricsLogger::~AsyncMetricsLogger()
{
    close();
}


bool AsyncMetricsLogger::open(
    const std::string& directory
)
{
    close();

    lastError_.clear();

    std::error_code error;

    std::filesystem::create_directories(
        directory,
        error
    );

    if (error) {

        lastError_ =
            "Cannot create async metrics directory: "
            +
            error.message();

        return false;
    }

    modelFile_.open(
        directory
        +
        "/async_model_metrics.csv",
        std::ios::out |
        std::ios::trunc
    );

    frameFile_.open(
        directory
        +
        "/async_frame_metrics.csv",
        std::ios::out |
        std::ios::trunc
    );

    if (!modelFile_ ||
        !frameFile_) {

        lastError_ =
            "Cannot open async metrics CSV files";

        close();

        return false;
    }

    modelFile_
        << "frame_id,"
        << "branch,"
        << "model,"
        << "worker_queue_wait_ms,"
        << "frame_age_ms,"
        << "preprocess_ms,"
        << "htp_wait_ms,"
        << "inference_ms,"
        << "postprocess_ms,"
        << "render_ms,"
        << "total_ms,"
        << "result_count\n";

    frameFile_
        << "frame_id,"
        << "capture_timestamp_us,"
        << "read_ms,"
        << "write_ms,"
        << "frame_age_ms,"
        << "render_queue_depth,"
        << "render_dropped_frames,"
        << "person_result_age_ms,"
        << "firesmoke_result_age_ms,"
        << "attendance_result_age_ms\n";

    return true;
}


void AsyncMetricsLogger::close()
{
    std::lock_guard<std::mutex> lock(
        mutex_
    );

    if (modelFile_.is_open()) {

        modelFile_.close();
    }

    if (frameFile_.is_open()) {

        frameFile_.close();
    }
}


bool AsyncMetricsLogger::logModel(
    const std::string& branch,
    const metrics::ModelMetrics& metric,
    double workerQueueWaitMs,
    double frameAgeMs
)
{
    std::lock_guard<std::mutex> lock(
        mutex_
    );

    if (!modelFile_.is_open()) {

        return false;
    }

    modelFile_
        << metric.frameId
        << ','
        << branch
        << ','
        << metric.model
        << ','
        << std::fixed
        << std::setprecision(6)
        << workerQueueWaitMs
        << ','
        << frameAgeMs
        << ','
        << metric.preprocessMs
        << ','
        << metric.queueMs
        << ','
        << metric.inferenceMs
        << ','
        << metric.postprocessMs
        << ','
        << metric.renderMs
        << ','
        << metric.totalMs
        << ','
        << metric.resultCount
        << '\n';

    modelFile_.flush();

    return
        static_cast<bool>(
            modelFile_
        );
}


bool AsyncMetricsLogger::logFrame(
    const AsyncFrameMetrics& metric
)
{
    std::lock_guard<std::mutex> lock(
        mutex_
    );

    if (!frameFile_.is_open()) {

        return false;
    }

    frameFile_
        << metric.frameId
        << ','
        << metric.captureTimestampUs
        << ','
        << std::fixed
        << std::setprecision(6)
        << metric.readMs
        << ','
        << metric.writeMs
        << ','
        << metric.frameAgeMs
        << ','
        << metric.renderQueueDepth
        << ','
        << metric.renderDroppedFrames
        << ','
        << metric.personResultAgeMs
        << ','
        << metric.fireSmokeResultAgeMs
        << ','
        << metric.attendanceResultAgeMs
        << '\n';

    frameFile_.flush();

    return
        static_cast<bool>(
            frameFile_
        );
}


const std::string&
AsyncMetricsLogger::lastError() const noexcept
{
    return
        lastError_;
}

} // namespace async_runtime