#pragma once

#include "metrics/metrics.hpp"

#include <string>

namespace benchmark {

class BenchmarkReport {
public:
    static bool generate(
        const std::string& metricsDirectory,
        const std::string& profile,
        const metrics::PipelineSummary& pipelineSummary,
        std::string& error
    );
};

} // namespace benchmark