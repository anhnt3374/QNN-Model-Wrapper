#pragma once

#include "video/frame.hpp"

#include <chrono>
#include <cstdint>
#include <memory>

namespace async_runtime {

struct AsyncFrame {
    using Clock =
        std::chrono::steady_clock;

    video::Frame frame;

    Clock::time_point publishedAt;

    double readMs =
        0.0;

    uint64_t sequence =
        0;
};

using AsyncFramePtr =
    std::shared_ptr<const AsyncFrame>;

} // namespace async_runtime