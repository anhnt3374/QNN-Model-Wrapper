#include "benchmark/benchmark_report.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace benchmark {

namespace {

struct ModelSample {
    double preprocessMs = 0.0;
    double queueMs = 0.0;
    double inferenceMs = 0.0;
    double postprocessMs = 0.0;
    double renderMs = 0.0;
    double totalMs = 0.0;
    double resultCount = 0.0;
};


struct ModelSummary {
    std::size_t samples = 0;

    double averagePreprocessMs = 0.0;
    double averageQueueMs = 0.0;
    double averageInferenceMs = 0.0;
    double averagePostprocessMs = 0.0;
    double averageRenderMs = 0.0;
    double averageTotalMs = 0.0;
    double averageResultCount = 0.0;

    double p50TotalMs = 0.0;
    double p95TotalMs = 0.0;
    double p99TotalMs = 0.0;
    double maxTotalMs = 0.0;
};


struct SystemSample {
    double elapsedMs = 0.0;
    double cpuPercent = 0.0;
    double rssMb = 0.0;
    double temperatureC = -1.0;
    double threadCount = 0.0;
};


struct SystemSummary {
    std::size_t steadySamples = 0;

    double averageCpuPercent = 0.0;

    double averageRssMb = 0.0;
    double peakRssMb = 0.0;

    double averageTemperatureC = -1.0;
    double peakTemperatureC = -1.0;

    double averageThreadCount = 0.0;
};


std::vector<std::string> splitCsv(
    const std::string& line
)
{
    std::vector<std::string> result;

    std::stringstream stream(
        line
    );

    std::string token;

    while (std::getline(
        stream,
        token,
        ','
    )) {

        result.push_back(
            token
        );
    }

    return result;
}


double average(
    const std::vector<double>& values
)
{
    if (values.empty()) {

        return 0.0;
    }

    double sum =
        0.0;

    for (const double value :
         values) {

        sum +=
            value;
    }

    return
        sum
        /
        static_cast<double>(
            values.size()
        );
}


double percentile(
    std::vector<double> values,
    double p
)
{
    if (values.empty()) {

        return 0.0;
    }

    std::sort(
        values.begin(),
        values.end()
    );

    if (values.size() == 1) {

        return
            values.front();
    }

    const double index =
        p
        *
        static_cast<double>(
            values.size() - 1
        );

    const std::size_t lower =
        static_cast<std::size_t>(
            std::floor(
                index
            )
        );

    const std::size_t upper =
        static_cast<std::size_t>(
            std::ceil(
                index
            )
        );

    if (lower == upper) {

        return
            values[lower];
    }

    const double fraction =
        index
        -
        static_cast<double>(
            lower
        );

    return
        values[lower]
        +
        (
            values[upper]
            -
            values[lower]
        )
        *
        fraction;
}


bool loadModelMetrics(
    const std::filesystem::path& path,
    std::map<
        std::string,
        ModelSummary
    >& summaries,
    std::string& error
)
{
    std::ifstream file(
        path
    );

    if (!file) {

        error =
            "Cannot open "
            +
            path.string();

        return false;
    }

    std::string line;

    // header
    if (!std::getline(
            file,
            line
        )) {

        error =
            "model_metrics.csv is empty";

        return false;
    }

    std::map<
        std::string,
        std::vector<ModelSample>
    > samples;

    while (std::getline(
        file,
        line
    )) {

        if (line.empty()) {

            continue;
        }

        const auto columns =
            splitCsv(
                line
            );

        // V3+ schema:
        //
        // frame_id
        // model
        // preprocess_ms
        // queue_ms
        // inference_ms
        // postprocess_ms
        // render_ms
        // total_ms
        // result_count

        if (columns.size() < 9) {

            continue;
        }

        try {

            ModelSample sample;

            sample.preprocessMs =
                std::stod(
                    columns[2]
                );

            sample.queueMs =
                std::stod(
                    columns[3]
                );

            sample.inferenceMs =
                std::stod(
                    columns[4]
                );

            sample.postprocessMs =
                std::stod(
                    columns[5]
                );

            sample.renderMs =
                std::stod(
                    columns[6]
                );

            sample.totalMs =
                std::stod(
                    columns[7]
                );

            sample.resultCount =
                std::stod(
                    columns[8]
                );

            samples[
                columns[1]
            ].push_back(
                sample
            );
        }
        catch (...) {

            continue;
        }
    }

    for (const auto& entry :
         samples) {

        const std::string& model =
            entry.first;

        const auto& modelSamples =
            entry.second;

        ModelSummary summary;

        summary.samples =
            modelSamples.size();

        std::vector<double> preprocess;
        std::vector<double> queue;
        std::vector<double> inference;
        std::vector<double> postprocess;
        std::vector<double> render;
        std::vector<double> total;
        std::vector<double> results;

        preprocess.reserve(
            modelSamples.size()
        );

        queue.reserve(
            modelSamples.size()
        );

        inference.reserve(
            modelSamples.size()
        );

        postprocess.reserve(
            modelSamples.size()
        );

        render.reserve(
            modelSamples.size()
        );

        total.reserve(
            modelSamples.size()
        );

        results.reserve(
            modelSamples.size()
        );

        for (const auto& sample :
             modelSamples) {

            preprocess.push_back(
                sample.preprocessMs
            );

            queue.push_back(
                sample.queueMs
            );

            inference.push_back(
                sample.inferenceMs
            );

            postprocess.push_back(
                sample.postprocessMs
            );

            render.push_back(
                sample.renderMs
            );

            total.push_back(
                sample.totalMs
            );

            results.push_back(
                sample.resultCount
            );
        }

        summary.averagePreprocessMs =
            average(
                preprocess
            );

        summary.averageQueueMs =
            average(
                queue
            );

        summary.averageInferenceMs =
            average(
                inference
            );

        summary.averagePostprocessMs =
            average(
                postprocess
            );

        summary.averageRenderMs =
            average(
                render
            );

        summary.averageTotalMs =
            average(
                total
            );

        summary.averageResultCount =
            average(
                results
            );

        summary.p50TotalMs =
            percentile(
                total,
                0.50
            );

        summary.p95TotalMs =
            percentile(
                total,
                0.95
            );

        summary.p99TotalMs =
            percentile(
                total,
                0.99
            );

        if (!total.empty()) {

            summary.maxTotalMs =
                *std::max_element(
                    total.begin(),
                    total.end()
                );
        }

        summaries[
            model
        ] =
            summary;
    }

    return true;
}


bool loadSystemMetrics(
    const std::filesystem::path& path,
    SystemSummary& summary,
    std::string& error
)
{
    std::ifstream file(
        path
    );

    if (!file) {

        error =
            "Cannot open "
            +
            path.string();

        return false;
    }

    std::string line;

    // header
    if (!std::getline(
            file,
            line
        )) {

        error =
            "system_metrics.csv is empty";

        return false;
    }

    std::vector<SystemSample> samples;

    while (std::getline(
        file,
        line
    )) {

        if (line.empty()) {

            continue;
        }

        const auto columns =
            splitCsv(
                line
            );

        if (columns.size() < 6) {

            continue;
        }

        try {

            SystemSample sample;

            sample.elapsedMs =
                std::stod(
                    columns[1]
                );

            sample.cpuPercent =
                std::stod(
                    columns[2]
                );

            sample.rssMb =
                std::stod(
                    columns[3]
                );

            sample.temperatureC =
                std::stod(
                    columns[4]
                );

            sample.threadCount =
                std::stod(
                    columns[5]
                );

            samples.push_back(
                sample
            );
        }
        catch (...) {

            continue;
        }
    }

    if (samples.empty()) {

        return true;
    }

    // PipelineProfiler forces one final sample during
    // teardown.
    //
    // That point frequently has much smaller RSS/CPU and
    // should not be mixed into steady-state averages.
    if (samples.size() > 1) {

        samples.pop_back();
    }

    if (samples.empty()) {

        return true;
    }

    summary.steadySamples =
        samples.size();

    std::vector<double> cpu;
    std::vector<double> rss;
    std::vector<double> temp;
    std::vector<double> threads;

    for (const auto& sample :
         samples) {

        cpu.push_back(
            sample.cpuPercent
        );

        rss.push_back(
            sample.rssMb
        );

        threads.push_back(
            sample.threadCount
        );

        if (sample.temperatureC >=
            0.0) {

            temp.push_back(
                sample.temperatureC
            );
        }
    }

    summary.averageCpuPercent =
        average(
            cpu
        );

    summary.averageRssMb =
        average(
            rss
        );

    summary.averageThreadCount =
        average(
            threads
        );

    if (!rss.empty()) {

        summary.peakRssMb =
            *std::max_element(
                rss.begin(),
                rss.end()
            );
    }

    if (!temp.empty()) {

        summary.averageTemperatureC =
            average(
                temp
            );

        summary.peakTemperatureC =
            *std::max_element(
                temp.begin(),
                temp.end()
            );
    }

    return true;
}


void writeNumber(
    std::ostream& output,
    double value
)
{
    output
        << std::fixed
        << std::setprecision(6)
        << value;
}

} // namespace


bool BenchmarkReport::generate(
    const std::string& metricsDirectory,
    const std::string& profile,
    const metrics::PipelineSummary& pipelineSummary,
    std::string& error
)
{
    error.clear();

    const std::filesystem::path root(
        metricsDirectory
    );

    std::map<
        std::string,
        ModelSummary
    > modelSummaries;

    SystemSummary systemSummary;

    if (!loadModelMetrics(
            root /
            "model_metrics.csv",
            modelSummaries,
            error
        )) {

        return false;
    }

    if (!loadSystemMetrics(
            root /
            "system_metrics.csv",
            systemSummary,
            error
        )) {

        return false;
    }

    std::ofstream output(
        root /
        "benchmark_summary.json",
        std::ios::out |
        std::ios::trunc
    );

    if (!output) {

        error =
            "Cannot create benchmark_summary.json";

        return false;
    }

    output
        << "{\n";

    output
        << "  \"profile\": \""
        << profile
        << "\",\n";

    // =====================================================
    // Pipeline
    // =====================================================

    output
        << "  \"pipeline\": {\n";

    output
        << "    \"frame_count\": "
        << pipelineSummary.frameCount
        << ",\n";

    output
        << "    \"runtime_ms\": ";

    writeNumber(
        output,
        pipelineSummary.runtimeMs
    );

    output
        << ",\n"
        << "    \"effective_fps\": ";

    writeNumber(
        output,
        pipelineSummary.effectiveFps
    );

    output
        << ",\n"
        << "    \"average_read_ms\": ";

    writeNumber(
        output,
        pipelineSummary.averageReadMs
    );

    output
        << ",\n"
        << "    \"average_process_ms\": ";

    writeNumber(
        output,
        pipelineSummary.averageProcessMs
    );

    output
        << ",\n"
        << "    \"average_write_ms\": ";

    writeNumber(
        output,
        pipelineSummary.averageWriteMs
    );

    output
        << ",\n"
        << "    \"average_total_ms\": ";

    writeNumber(
        output,
        pipelineSummary.averageTotalMs
    );

    output
        << ",\n"
        << "    \"p50_total_ms\": ";

    writeNumber(
        output,
        pipelineSummary.p50TotalMs
    );

    output
        << ",\n"
        << "    \"p95_total_ms\": ";

    writeNumber(
        output,
        pipelineSummary.p95TotalMs
    );

    output
        << ",\n"
        << "    \"p99_total_ms\": ";

    writeNumber(
        output,
        pipelineSummary.p99TotalMs
    );

    output
        << ",\n"
        << "    \"max_total_ms\": ";

    writeNumber(
        output,
        pipelineSummary.maxTotalMs
    );

    output
        << "\n"
        << "  },\n";

    // =====================================================
    // Models
    // =====================================================

    output
        << "  \"models\": {\n";

    std::size_t modelIndex =
        0;

    for (const auto& entry :
         modelSummaries) {

        const auto& name =
            entry.first;

        const auto& summary =
            entry.second;

        output
            << "    \""
            << name
            << "\": {\n";

        output
            << "      \"samples\": "
            << summary.samples
            << ",\n";

        output
            << "      \"average_preprocess_ms\": ";

        writeNumber(
            output,
            summary.averagePreprocessMs
        );

        output
            << ",\n"
            << "      \"average_queue_ms\": ";

        writeNumber(
            output,
            summary.averageQueueMs
        );

        output
            << ",\n"
            << "      \"average_inference_ms\": ";

        writeNumber(
            output,
            summary.averageInferenceMs
        );

        output
            << ",\n"
            << "      \"average_postprocess_ms\": ";

        writeNumber(
            output,
            summary.averagePostprocessMs
        );

        output
            << ",\n"
            << "      \"average_render_ms\": ";

        writeNumber(
            output,
            summary.averageRenderMs
        );

        output
            << ",\n"
            << "      \"average_total_ms\": ";

        writeNumber(
            output,
            summary.averageTotalMs
        );

        output
            << ",\n"
            << "      \"average_result_count\": ";

        writeNumber(
            output,
            summary.averageResultCount
        );

        output
            << ",\n"
            << "      \"p50_total_ms\": ";

        writeNumber(
            output,
            summary.p50TotalMs
        );

        output
            << ",\n"
            << "      \"p95_total_ms\": ";

        writeNumber(
            output,
            summary.p95TotalMs
        );

        output
            << ",\n"
            << "      \"p99_total_ms\": ";

        writeNumber(
            output,
            summary.p99TotalMs
        );

        output
            << ",\n"
            << "      \"max_total_ms\": ";

        writeNumber(
            output,
            summary.maxTotalMs
        );

        output
            << "\n"
            << "    }";

        ++modelIndex;

        if (modelIndex <
            modelSummaries.size()) {

            output
                << ",";
        }

        output
            << "\n";
    }

    output
        << "  },\n";

    // =====================================================
    // System
    // =====================================================

    output
        << "  \"system\": {\n";

    output
        << "    \"steady_state_samples\": "
        << systemSummary.steadySamples
        << ",\n";

    output
        << "    \"average_cpu_percent\": ";

    writeNumber(
        output,
        systemSummary.averageCpuPercent
    );

    output
        << ",\n"
        << "    \"average_rss_mb\": ";

    writeNumber(
        output,
        systemSummary.averageRssMb
    );

    output
        << ",\n"
        << "    \"peak_rss_mb\": ";

    writeNumber(
        output,
        systemSummary.peakRssMb
    );

    output
        << ",\n"
        << "    \"average_temperature_c\": ";

    writeNumber(
        output,
        systemSummary.averageTemperatureC
    );

    output
        << ",\n"
        << "    \"peak_temperature_c\": ";

    writeNumber(
        output,
        systemSummary.peakTemperatureC
    );

    output
        << ",\n"
        << "    \"average_thread_count\": ";

    writeNumber(
        output,
        systemSummary.averageThreadCount
    );

    output
        << "\n"
        << "  }\n";

    output
        << "}\n";

    if (!output) {

        error =
            "Failed writing benchmark_summary.json";

        return false;
    }

    return true;
}

} // namespace benchmark