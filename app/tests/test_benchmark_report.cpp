#include "benchmark/benchmark_report.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

bool contains(
    const std::string& text,
    const std::string& value
)
{
    return
        text.find(
            value
        )
        !=
        std::string::npos;
}

} // namespace


int main()
{
    namespace fs =
        std::filesystem;

    const fs::path root =
        fs::temp_directory_path()
        /
        "qcs6490_benchmark_report_test";

    fs::remove_all(
        root
    );

    fs::create_directories(
        root
    );

    // =====================================================
    // model_metrics.csv
    // =====================================================

    {
        std::ofstream file(
            root /
            "model_metrics.csv"
        );

        file
            << "frame_id,model,preprocess_ms,queue_ms,"
               "inference_ms,postprocess_ms,render_ms,"
               "total_ms,result_count\n";

        file
            << "0,person,10,0,12,1,2,25,1\n";

        file
            << "3,person,20,0,14,1,2,37,2\n";

        file
            << "0,firesmoke,5,0,8,2,1,16,3\n";
    }

    // =====================================================
    // system_metrics.csv
    //
    // Last row is teardown and must be ignored.
    // =====================================================

    {
        std::ofstream file(
            root /
            "system_metrics.csv"
        );

        file
            << "sample_index,elapsed_ms,cpu_percent,"
               "rss_mb,temperature_c,thread_count\n";

        file
            << "0,1000,200,300,60,32\n";

        file
            << "1,2000,220,320,62,32\n";

        file
            << "2,2100,50,100,55,10\n";
    }

    metrics::PipelineSummary pipeline;

    pipeline.frameCount =
        100;

    pipeline.runtimeMs =
        5000.0;

    pipeline.effectiveFps =
        20.0;

    pipeline.averageReadMs =
        1.5;

    pipeline.averageProcessMs =
        40.0;

    pipeline.averageWriteMs =
        3.0;

    pipeline.averageTotalMs =
        44.5;

    pipeline.p50TotalMs =
        10.0;

    pipeline.p95TotalMs =
        80.0;

    pipeline.p99TotalMs =
        90.0;

    pipeline.maxTotalMs =
        100.0;

    std::string error;

    if (!benchmark::BenchmarkReport::generate(
            root.string(),
            "b2",
            pipeline,
            error
        )) {

        std::cerr
            << "[FAIL] "
            << error
            << '\n';

        return EXIT_FAILURE;
    }

    std::ifstream reportFile(
        root /
        "benchmark_summary.json"
    );

    std::stringstream buffer;

    buffer
        << reportFile.rdbuf();

    const std::string report =
        buffer.str();

    if (!contains(
            report,
            "\"profile\": \"b2\""
        )) {

        std::cerr
            << "[FAIL] profile missing\n";

        return EXIT_FAILURE;
    }

    if (!contains(
            report,
            "\"person\""
        )) {

        std::cerr
            << "[FAIL] person summary missing\n";

        return EXIT_FAILURE;
    }

    if (!contains(
            report,
            "\"firesmoke\""
        )) {

        std::cerr
            << "[FAIL] FireSmoke summary missing\n";

        return EXIT_FAILURE;
    }

    // Person average total:
    //
    // (25 + 37) / 2 = 31
    if (!contains(
            report,
            "\"average_total_ms\": 31.000000"
        )) {

        std::cerr
            << "[FAIL] model average incorrect\n";

        return EXIT_FAILURE;
    }

    // Last system sample excluded:
    //
    // CPU = (200 + 220) / 2 = 210
    if (!contains(
            report,
            "\"average_cpu_percent\": 210.000000"
        )) {

        std::cerr
            << "[FAIL] teardown exclusion incorrect\n";

        return EXIT_FAILURE;
    }

    // RSS:
    //
    // (300 + 320) / 2 = 310
    if (!contains(
            report,
            "\"average_rss_mb\": 310.000000"
        )) {

        std::cerr
            << "[FAIL] RSS average incorrect\n";

        return EXIT_FAILURE;
    }

    std::cout
        << "[PASS] benchmark report generated\n";

    std::cout
        << "[PASS] model summaries valid\n";

    std::cout
        << "[PASS] teardown system sample excluded\n";

    std::cout
        << "[PASS] V7 benchmark report test complete\n";

    fs::remove_all(
        root
    );

    return EXIT_SUCCESS;
}