#include "async_runtime/htp_execution_gate.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

int main()
{
    using namespace std::chrono_literals;

    async_runtime::HtpExecutionGate gate;

    std::atomic<int> inside{
        0
    };

    std::atomic<int> maxInside{
        0
    };

    auto inference =
        [&]() {

            double waitMs =
                0.0;

            double executionMs =
                0.0;

            return gate.execute(
                [&]() {

                    const int now =
                        ++inside;

                    int previous =
                        maxInside.load();

                    while (
                        now > previous &&
                        !maxInside.compare_exchange_weak(
                            previous,
                            now
                        )
                    ) {
                    }

                    std::this_thread::sleep_for(
                        25ms
                    );

                    --inside;

                    return true;
                },
                waitMs,
                executionMs
            );
        };

    std::thread first(
        inference
    );

    std::thread second(
        inference
    );

    first.join();

    second.join();

    if (maxInside.load() != 1) {

        std::cerr
            << "[FAIL] multiple HTP executions overlapped\n";

        return EXIT_FAILURE;
    }

    const auto stats =
        gate.stats();

    if (stats.executionCount != 2) {

        std::cerr
            << "[FAIL] execution counter invalid\n";

        return EXIT_FAILURE;
    }

    if (stats.totalExecutionMs <
        40.0) {

        std::cerr
            << "[FAIL] execution timing invalid\n";

        return EXIT_FAILURE;
    }

    if (stats.totalWaitMs <=
        0.0) {

        std::cerr
            << "[FAIL] expected one caller to wait\n";

        return EXIT_FAILURE;
    }

    std::cout
        << "[PASS] HTP execution serialized\n";

    std::cout
        << "[INFO] total wait: "
        << stats.totalWaitMs
        << " ms\n";

    std::cout
        << "[INFO] max wait: "
        << stats.maxWaitMs
        << " ms\n";

    std::cout
        << "[PASS] V8 HTP gate test complete\n";

    return EXIT_SUCCESS;
}