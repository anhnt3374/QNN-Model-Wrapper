#pragma once

#include "metrics/metrics.hpp"

#include <fstream>
#include <string>

namespace metrics {

class MetricsLogger {
public:
    MetricsLogger() = default;

    ~MetricsLogger();

    MetricsLogger(
        const MetricsLogger&
    ) = delete;

    MetricsLogger& operator=(
        const MetricsLogger&
    ) = delete;

    bool open(
        const std::string& directory
    );

    void close();

    bool writeFrame(
        const FrameMetrics& metrics
    );

    bool writeModel(
        const ModelMetrics& metrics
    );

    bool writeSystem(
        const SystemMetrics& metrics
    );

    void flush();

    bool isOpen() const noexcept;

    const std::string&
    directory() const noexcept;

    const std::string&
    lastError() const noexcept;

private:
    std::string directory_;

    std::ofstream frameFile_;

    std::ofstream modelFile_;

    std::ofstream systemFile_;

    std::string lastError_;
};

} // namespace metrics