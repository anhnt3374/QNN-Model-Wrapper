#include "metrics/profiler.hpp"

#include <sys/resource.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>

namespace metrics {

PipelineProfiler::PipelineProfiler(
    double systemSampleIntervalMs
)
    : systemSampleIntervalMs_(
          systemSampleIntervalMs
      )
{
    if (!std::isfinite(
            systemSampleIntervalMs_
        ) ||
        systemSampleIntervalMs_ <= 0.0) {

        systemSampleIntervalMs_ =
            1000.0;
    }
}

void PipelineProfiler::reset()
{
    running_ =
        false;

    runStart_ = {};

    runEnd_ = {};

    lastSystemSample_ = {};

    lastCpuSeconds_ =
        0.0;

    nextSystemSampleIndex_ =
        0;

    readLatencies_.clear();

    processLatencies_.clear();

    writeLatencies_.clear();

    totalLatencies_.clear();
}

void PipelineProfiler::startRun()
{
    reset();

    running_ =
        true;

    runStart_ =
        Clock::now();

    runEnd_ =
        runStart_;

    lastSystemSample_ =
        runStart_;

    lastCpuSeconds_ =
        processCpuSeconds();
}

void PipelineProfiler::finishRun()
{
    if (!running_) {
        return;
    }

    runEnd_ =
        Clock::now();

    running_ =
        false;
}

void PipelineProfiler::recordFrame(
    const FrameMetrics& metrics
)
{
    readLatencies_.push_back(
        metrics.readMs
    );

    processLatencies_.push_back(
        metrics.processMs
    );

    writeLatencies_.push_back(
        metrics.writeMs
    );

    totalLatencies_.push_back(
        metrics.totalMs
    );
}

bool PipelineProfiler::sampleSystemIfDue(
    SystemMetrics& output,
    bool force
)
{
    if (runStart_ ==
        Clock::time_point{}) {

        return false;
    }

    const Clock::time_point now =
        Clock::now();

    const double elapsedSinceSampleMs =
        durationMs(
            lastSystemSample_,
            now
        );

    if (!force &&
        elapsedSinceSampleMs <
            systemSampleIntervalMs_) {

        return false;
    }

    const double currentCpuSeconds =
        processCpuSeconds();

    const double wallSeconds =
        elapsedSinceSampleMs
        /
        1000.0;

    double cpuPercent =
        0.0;

    if (wallSeconds > 0.0) {

        const double cpuDeltaSeconds =
            currentCpuSeconds
            -
            lastCpuSeconds_;

        cpuPercent =
            (
                cpuDeltaSeconds
                /
                wallSeconds
            )
            *
            100.0;

        if (!std::isfinite(
                cpuPercent
            ) ||
            cpuPercent < 0.0) {

            cpuPercent =
                0.0;
        }
    }

    output = {};

    output.sampleIndex =
        nextSystemSampleIndex_++;

    output.elapsedMs =
        durationMs(
            runStart_,
            now
        );

    output.cpuPercent =
        cpuPercent;

    output.rssMb =
        processRssMb();

    output.temperatureC =
        maxTemperatureC();

    output.threadCount =
        processThreadCount();

    lastSystemSample_ =
        now;

    lastCpuSeconds_ =
        currentCpuSeconds;

    return true;
}

PipelineSummary
PipelineProfiler::summary() const
{
    PipelineSummary result;

    result.frameCount =
        totalLatencies_.size();

    if (runStart_ !=
        Clock::time_point{}) {

        const Clock::time_point end =
            running_
                ? Clock::now()
                : runEnd_;

        result.runtimeMs =
            durationMs(
                runStart_,
                end
            );
    }

    if (result.runtimeMs > 0.0) {

        result.effectiveFps =
            (
                static_cast<double>(
                    result.frameCount
                )
                *
                1000.0
            )
            /
            result.runtimeMs;
    }

    result.averageReadMs =
        average(
            readLatencies_
        );

    result.averageProcessMs =
        average(
            processLatencies_
        );

    result.averageWriteMs =
        average(
            writeLatencies_
        );

    result.averageTotalMs =
        average(
            totalLatencies_
        );

    result.p50TotalMs =
        percentile(
            totalLatencies_,
            50.0
        );

    result.p95TotalMs =
        percentile(
            totalLatencies_,
            95.0
        );

    result.p99TotalMs =
        percentile(
            totalLatencies_,
            99.0
        );

    if (!totalLatencies_.empty()) {

        result.maxTotalMs =
            *std::max_element(
                totalLatencies_.begin(),
                totalLatencies_.end()
            );
    }

    return result;
}

uint64_t
PipelineProfiler::frameCount() const noexcept
{
    return
        totalLatencies_.size();
}

double PipelineProfiler::durationMs(
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

double PipelineProfiler::average(
    const std::vector<double>& values
)
{
    if (values.empty()) {
        return 0.0;
    }

    const double total =
        std::accumulate(
            values.begin(),
            values.end(),
            0.0
        );

    return
        total
        /
        static_cast<double>(
            values.size()
        );
}

double PipelineProfiler::percentile(
    const std::vector<double>& values,
    double percentileValue
)
{
    if (values.empty()) {
        return 0.0;
    }

    std::vector<double> sorted =
        values;

    std::sort(
        sorted.begin(),
        sorted.end()
    );

    if (percentileValue <= 0.0) {

        return
            sorted.front();
    }

    if (percentileValue >= 100.0) {

        return
            sorted.back();
    }

    const double position =
        (
            percentileValue
            /
            100.0
        )
        *
        static_cast<double>(
            sorted.size() - 1
        );

    const auto lower =
        static_cast<std::size_t>(
            std::floor(
                position
            )
        );

    const auto upper =
        static_cast<std::size_t>(
            std::ceil(
                position
            )
        );

    if (lower == upper) {

        return
            sorted[lower];
    }

    const double fraction =
        position
        -
        static_cast<double>(
            lower
        );

    return
        sorted[lower]
        +
        (
            sorted[upper]
            -
            sorted[lower]
        )
        *
        fraction;
}

double PipelineProfiler::processCpuSeconds()
{
    struct rusage usage {};

    if (getrusage(
            RUSAGE_SELF,
            &usage
        ) != 0) {

        return 0.0;
    }

    const double userSeconds =
        static_cast<double>(
            usage.ru_utime.tv_sec
        )
        +
        static_cast<double>(
            usage.ru_utime.tv_usec
        )
        /
        1'000'000.0;

    const double systemSeconds =
        static_cast<double>(
            usage.ru_stime.tv_sec
        )
        +
        static_cast<double>(
            usage.ru_stime.tv_usec
        )
        /
        1'000'000.0;

    return
        userSeconds
        +
        systemSeconds;
}

double PipelineProfiler::processRssMb()
{
    std::ifstream file(
        "/proc/self/status"
    );

    if (!file) {

        return 0.0;
    }

    std::string key;

    while (file >> key) {

        if (key == "VmRSS:") {

            double valueKb =
                0.0;

            std::string unit;

            file
                >> valueKb
                >> unit;

            return
                valueKb
                /
                1024.0;
        }

        std::string rest;

        std::getline(
            file,
            rest
        );
    }

    return 0.0;
}

int PipelineProfiler::processThreadCount()
{
    std::ifstream file(
        "/proc/self/status"
    );

    if (!file) {

        return 0;
    }

    std::string key;

    while (file >> key) {

        if (key == "Threads:") {

            int count =
                0;

            file >> count;

            return
                count;
        }

        std::string rest;

        std::getline(
            file,
            rest
        );
    }

    return 0;
}

double PipelineProfiler::maxTemperatureC()
{
    namespace fs =
        std::filesystem;

    const fs::path root(
        "/sys/class/thermal"
    );

    std::error_code error;

    if (!fs::exists(
            root,
            error
        )) {

        return -1.0;
    }

    double maxTemperature =
        -1.0;

    fs::directory_iterator iterator(
        root,
        error
    );

    if (error) {

        return -1.0;
    }

    for (const auto& entry :
         iterator) {

        const std::string name =
            entry
                .path()
                .filename()
                .string();

        if (name.rfind(
                "thermal_zone",
                0
            ) != 0) {

            continue;
        }

        std::ifstream file(
            entry.path()
            /
            "temp"
        );

        if (!file) {
            continue;
        }

        double rawTemperature =
            0.0;

        file
            >> rawTemperature;

        if (!file) {
            continue;
        }

        // Linux thermal zones normally expose millidegrees.
        //
        // Some platforms may already expose Celsius.
        double temperatureC =
            rawTemperature;

        if (rawTemperature > 1000.0) {

            temperatureC =
                rawTemperature
                /
                1000.0;
        }

        if (temperatureC <
                -50.0 ||
            temperatureC >
                200.0) {

            continue;
        }

        maxTemperature =
            std::max(
                maxTemperature,
                temperatureC
            );
    }

    return
        maxTemperature;
}

} // namespace metrics