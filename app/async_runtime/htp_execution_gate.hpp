#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <utility>

namespace async_runtime {

class HtpExecutionGate {
public:
    using Clock =
        std::chrono::steady_clock;

    struct Stats {
        uint64_t executionCount =
            0;

        double totalWaitMs =
            0.0;

        double maxWaitMs =
            0.0;

        double totalExecutionMs =
            0.0;

        double maxExecutionMs =
            0.0;
    };


    HtpExecutionGate() = default;

    HtpExecutionGate(
        const HtpExecutionGate&
    ) = delete;

    HtpExecutionGate& operator=(
        const HtpExecutionGate&
    ) = delete;


    // =====================================================
    // Serialize ONLY the inference call.
    //
    // Preprocess happens BEFORE this method.
    // Postprocess happens AFTER this method.
    // =====================================================

    template <typename Function>
    bool execute(
        Function&& function,
        double& waitMs,
        double& executionMs
    )
    {
        const Clock::time_point waitStart =
            Clock::now();

        std::unique_lock<std::mutex> executionLock(
            executionMutex_
        );

        const Clock::time_point executionStart =
            Clock::now();

        waitMs =
            durationMs(
                waitStart,
                executionStart
            );

        const bool success =
            std::forward<Function>(
                function
            )();

        const Clock::time_point executionEnd =
            Clock::now();

        executionMs =
            durationMs(
                executionStart,
                executionEnd
            );

        executionLock.unlock();

        {
            std::lock_guard<std::mutex> statsLock(
                statsMutex_
            );

            ++stats_.executionCount;

            stats_.totalWaitMs +=
                waitMs;

            stats_.totalExecutionMs +=
                executionMs;

            if (waitMs >
                stats_.maxWaitMs) {

                stats_.maxWaitMs =
                    waitMs;
            }

            if (executionMs >
                stats_.maxExecutionMs) {

                stats_.maxExecutionMs =
                    executionMs;
            }
        }

        return success;
    }


    Stats stats() const
    {
        std::lock_guard<std::mutex> lock(
            statsMutex_
        );

        return
            stats_;
    }


    void resetStats()
    {
        std::lock_guard<std::mutex> lock(
            statsMutex_
        );

        stats_ = {};
    }


private:
    static double durationMs(
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


private:
    // Only one QNN graph executes inside this mutex.
    std::mutex executionMutex_;

    mutable std::mutex statsMutex_;

    Stats stats_;
};

} // namespace async_runtime