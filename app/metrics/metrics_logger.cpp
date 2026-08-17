#include "metrics/metrics_logger.hpp"

#include <filesystem>
#include <iomanip>

namespace metrics {

MetricsLogger::~MetricsLogger()
{
    close();
}

bool MetricsLogger::open(
    const std::string& directory
)
{
    close();

    lastError_.clear();

    if (directory.empty()) {

        lastError_ =
            "Metrics directory is empty";

        return false;
    }

    namespace fs =
        std::filesystem;

    std::error_code error;

    fs::create_directories(
        directory,
        error
    );

    if (error) {

        lastError_ =
            "Cannot create metrics directory: "
            +
            directory
            +
            ": "
            +
            error.message();

        return false;
    }

    directory_ =
        directory;

    const fs::path root(
        directory_
    );

    frameFile_.open(
        root / "frame_metrics.csv",
        std::ios::out |
        std::ios::trunc
    );

    if (!frameFile_) {

        lastError_ =
            "Cannot open frame_metrics.csv";

        close();

        return false;
    }

    modelFile_.open(
        root / "model_metrics.csv",
        std::ios::out |
        std::ios::trunc
    );

    if (!modelFile_) {

        lastError_ =
            "Cannot open model_metrics.csv";

        close();

        return false;
    }

    systemFile_.open(
        root / "system_metrics.csv",
        std::ios::out |
        std::ios::trunc
    );

    if (!systemFile_) {

        lastError_ =
            "Cannot open system_metrics.csv";

        close();

        return false;
    }

    frameFile_
        << "frame_id,"
        << "capture_timestamp_us,"
        << "read_ms,"
        << "process_ms,"
        << "write_ms,"
        << "total_ms,"
        << "instantaneous_fps"
        << '\n';

    modelFile_
        << "frame_id,"
        << "model,"
        << "preprocess_ms,"
        << "queue_ms,"
        << "inference_ms,"
        << "postprocess_ms,"
        << "render_ms,"
        << "total_ms,"
        << "result_count"
        << '\n';

    systemFile_
        << "sample_index,"
        << "elapsed_ms,"
        << "cpu_percent,"
        << "rss_mb,"
        << "temperature_c,"
        << "thread_count"
        << '\n';

    flush();

    return true;
}

void MetricsLogger::close()
{
    if (frameFile_.is_open()) {
        frameFile_.close();
    }

    if (modelFile_.is_open()) {
        modelFile_.close();
    }

    if (systemFile_.is_open()) {
        systemFile_.close();
    }
}

bool MetricsLogger::writeFrame(
    const FrameMetrics& metrics
)
{
    if (!frameFile_.is_open()) {

        lastError_ =
            "frame_metrics.csv is not open";

        return false;
    }

    frameFile_
        << metrics.frameId
        << ','
        << metrics.captureTimestampUs
        << ','
        << std::fixed
        << std::setprecision(6)
        << metrics.readMs
        << ','
        << metrics.processMs
        << ','
        << metrics.writeMs
        << ','
        << metrics.totalMs
        << ','
        << metrics.instantaneousFps
        << '\n';

    if (!frameFile_) {

        lastError_ =
            "Failed writing frame_metrics.csv";

        return false;
    }

    return true;
}

bool MetricsLogger::writeModel(
    const ModelMetrics& metrics
)
{
    if (!modelFile_.is_open()) {

        lastError_ =
            "model_metrics.csv is not open";

        return false;
    }

    modelFile_
        << metrics.frameId
        << ','
        << metrics.model
        << ','
        << std::fixed
        << std::setprecision(6)
        << metrics.preprocessMs
        << ','
        << metrics.queueMs
        << ','
        << metrics.inferenceMs
        << ','
        << metrics.postprocessMs
        << ','
        << metrics.renderMs
        << ','
        << metrics.totalMs
        << ','
        << metrics.resultCount
        << '\n';

    if (!modelFile_) {

        lastError_ =
            "Failed writing model_metrics.csv";

        return false;
    }

    return true;
}

bool MetricsLogger::writeSystem(
    const SystemMetrics& metrics
)
{
    if (!systemFile_.is_open()) {

        lastError_ =
            "system_metrics.csv is not open";

        return false;
    }

    systemFile_
        << metrics.sampleIndex
        << ','
        << std::fixed
        << std::setprecision(6)
        << metrics.elapsedMs
        << ','
        << metrics.cpuPercent
        << ','
        << metrics.rssMb
        << ','
        << metrics.temperatureC
        << ','
        << metrics.threadCount
        << '\n';

    if (!systemFile_) {

        lastError_ =
            "Failed writing system_metrics.csv";

        return false;
    }

    return true;
}

void MetricsLogger::flush()
{
    if (frameFile_.is_open()) {
        frameFile_.flush();
    }

    if (modelFile_.is_open()) {
        modelFile_.flush();
    }

    if (systemFile_.is_open()) {
        systemFile_.flush();
    }
}

bool MetricsLogger::isOpen() const noexcept
{
    return
        frameFile_.is_open()
        &&
        modelFile_.is_open()
        &&
        systemFile_.is_open();
}

const std::string&
MetricsLogger::directory() const noexcept
{
    return
        directory_;
}

const std::string&
MetricsLogger::lastError() const noexcept
{
    return
        lastError_;
}

} // namespace metrics