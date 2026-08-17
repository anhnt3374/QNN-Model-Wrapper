#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <mutex>
#include <utility>

namespace async_runtime {

struct OverlayResult {
    uint64_t sourceFrameId =
        0;

    int64_t sourceTimestampUs =
        0;

    cv::Mat overlay;

    double processingMs =
        0.0;

    bool valid =
        false;
};


class OverlayResultCache {
public:
    void update(
        OverlayResult result
    )
    {
        std::lock_guard<std::mutex> lock(
            mutex_
        );

        latest_ =
            std::move(
                result
            );

        latest_.valid =
            true;

        ++updateCount_;
    }


    bool latest(
        OverlayResult& result
    ) const
    {
        std::lock_guard<std::mutex> lock(
            mutex_
        );

        if (!latest_.valid) {

            return false;
        }

        result =
            latest_;

        // Explicit clone so renderer owns a stable snapshot
        // after releasing the mutex.
        if (!latest_.overlay.empty()) {

            result.overlay =
                latest_.overlay.clone();
        }

        return true;
    }


    void clear()
    {
        std::lock_guard<std::mutex> lock(
            mutex_
        );

        latest_ = {};

        updateCount_ =
            0;
    }


    uint64_t updateCount() const
    {
        std::lock_guard<std::mutex> lock(
            mutex_
        );

        return
            updateCount_;
    }


private:
    mutable std::mutex mutex_;

    OverlayResult latest_;

    uint64_t updateCount_ =
        0;
};

} // namespace async_runtime