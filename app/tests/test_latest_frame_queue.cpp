#include "async_runtime/latest_frame_queue.hpp"

#include <cstdlib>
#include <iostream>

namespace {

bool require(
    bool condition,
    const char* message
)
{
    if (!condition) {

        std::cerr
            << "[FAIL] "
            << message
            << '\n';

        return false;
    }

    std::cout
        << "[PASS] "
        << message
        << '\n';

    return true;
}

} // namespace


int main()
{
    async_runtime::LatestFrameQueue<int> queue(
        2
    );

    // =====================================================
    // Keep latest / drop oldest
    // =====================================================

    queue.push(
        100
    );

    queue.push(
        101
    );

    queue.push(
        102
    );

    if (!require(
            queue.size() == 2,
            "queue depth stays bounded"
        )) {

        return EXIT_FAILURE;
    }

    if (!require(
            queue.droppedCount() == 1,
            "oldest frame is dropped"
        )) {

        return EXIT_FAILURE;
    }

    int value =
        0;

    if (!queue.waitPop(
            value
        )) {

        std::cerr
            << "[FAIL] cannot pop\n";

        return EXIT_FAILURE;
    }

    if (!require(
            value == 101,
            "frame 100 was dropped"
        )) {

        return EXIT_FAILURE;
    }

    if (!queue.waitPop(
            value
        )) {

        return EXIT_FAILURE;
    }

    if (!require(
            value == 102,
            "newest frame remains"
        )) {

        return EXIT_FAILURE;
    }

    // =====================================================
    // Explicit pop-latest
    // =====================================================

    queue.push(
        200
    );

    queue.push(
        201
    );

    if (!queue.waitPopLatest(
            value
        )) {

        return EXIT_FAILURE;
    }

    if (!require(
            value == 201,
            "waitPopLatest returns newest item"
        )) {

        return EXIT_FAILURE;
    }

    if (!require(
            queue.size() == 0,
            "waitPopLatest clears stale items"
        )) {

        return EXIT_FAILURE;
    }

    // =====================================================
    // Close wakes / terminates consumers
    // =====================================================

    queue.close();

    if (!require(
            !queue.push(
                300
            ),
            "closed queue rejects new pushes"
        )) {

        return EXIT_FAILURE;
    }

    if (!require(
            !queue.waitPop(
                value
            ),
            "closed empty queue terminates pop"
        )) {

        return EXIT_FAILURE;
    }

    std::cout
        << "[PASS] V8 LatestFrameQueue test complete\n";

    return EXIT_SUCCESS;
}