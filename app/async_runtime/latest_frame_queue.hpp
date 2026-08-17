#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <utility>

namespace async_runtime {

template <typename T>
class LatestFrameQueue {
public:
    explicit LatestFrameQueue(
        std::size_t capacity = 2
    )
        : capacity_(
              capacity == 0
                  ? 1
                  : capacity
          )
    {
    }

    LatestFrameQueue(
        const LatestFrameQueue&
    ) = delete;

    LatestFrameQueue& operator=(
        const LatestFrameQueue&
    ) = delete;

    // =====================================================
    // Push newest item.
    //
    // If the queue is full:
    //
    // DROP THE OLDEST ITEM.
    //
    // Example capacity=2:
    //
    // [101, 102] + 103
    //
    // becomes:
    //
    // [102, 103]
    // =====================================================

    bool push(
        T value
    )
    {
        std::unique_lock<std::mutex> lock(
            mutex_
        );

        if (closed_) {

            return false;
        }

        if (queue_.size() >=
            capacity_) {

            queue_.pop_front();

            ++droppedCount_;
        }

        queue_.push_back(
            std::move(
                value
            )
        );

        ++pushedCount_;

        const std::size_t currentDepth =
            queue_.size();

        if (currentDepth >
            maxDepth_) {

            maxDepth_ =
                currentDepth;
        }

        lock.unlock();

        condition_.notify_one();

        return true;
    }


    // =====================================================
    // Pop NEXT available item.
    //
    // This preserves the queue ordering but because push()
    // already drops old items, backlog is bounded.
    // =====================================================

    bool waitPop(
        T& value
    )
    {
        std::unique_lock<std::mutex> lock(
            mutex_
        );

        condition_.wait(
            lock,
            [this]() {

                return
                    closed_
                    ||
                    !queue_.empty();
            }
        );

        if (queue_.empty()) {

            return false;
        }

        value =
            std::move(
                queue_.front()
            );

        queue_.pop_front();

        ++poppedCount_;

        return true;
    }


    // =====================================================
    // Pop newest and discard everything older.
    //
    // Useful if a worker becomes significantly behind.
    // =====================================================

    bool waitPopLatest(
        T& value
    )
    {
        std::unique_lock<std::mutex> lock(
            mutex_
        );

        condition_.wait(
            lock,
            [this]() {

                return
                    closed_
                    ||
                    !queue_.empty();
            }
        );

        if (queue_.empty()) {

            return false;
        }

        if (queue_.size() > 1) {

            droppedCount_ +=
                queue_.size() - 1;
        }

        value =
            std::move(
                queue_.back()
            );

        queue_.clear();

        ++poppedCount_;

        return true;
    }


    void close()
    {
        {
            std::lock_guard<std::mutex> lock(
                mutex_
            );

            closed_ =
                true;
        }

        condition_.notify_all();
    }


    void reset()
    {
        std::lock_guard<std::mutex> lock(
            mutex_
        );

        queue_.clear();

        closed_ =
            false;

        pushedCount_ =
            0;

        poppedCount_ =
            0;

        droppedCount_ =
            0;

        maxDepth_ =
            0;
    }


    bool closed() const
    {
        std::lock_guard<std::mutex> lock(
            mutex_
        );

        return
            closed_;
    }


    std::size_t size() const
    {
        std::lock_guard<std::mutex> lock(
            mutex_
        );

        return
            queue_.size();
    }


    std::size_t capacity() const noexcept
    {
        return
            capacity_;
    }


    uint64_t pushedCount() const
    {
        std::lock_guard<std::mutex> lock(
            mutex_
        );

        return
            pushedCount_;
    }


    uint64_t poppedCount() const
    {
        std::lock_guard<std::mutex> lock(
            mutex_
        );

        return
            poppedCount_;
    }


    uint64_t droppedCount() const
    {
        std::lock_guard<std::mutex> lock(
            mutex_
        );

        return
            droppedCount_;
    }


    std::size_t maxDepth() const
    {
        std::lock_guard<std::mutex> lock(
            mutex_
        );

        return
            maxDepth_;
    }

private:
    const std::size_t capacity_;

    mutable std::mutex mutex_;

    std::condition_variable condition_;

    std::deque<T> queue_;

    bool closed_ =
        false;

    uint64_t pushedCount_ =
        0;

    uint64_t poppedCount_ =
        0;

    uint64_t droppedCount_ =
        0;

    std::size_t maxDepth_ =
        0;
};

} // namespace async_runtime