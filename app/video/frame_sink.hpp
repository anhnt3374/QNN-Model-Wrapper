#pragma once

#include "video/frame.hpp"

#include <string>

namespace video {

class FrameSink {
public:
    virtual ~FrameSink() = default;

    FrameSink(
        const FrameSink&
    ) = delete;

    FrameSink& operator=(
        const FrameSink&
    ) = delete;

    virtual bool open(
        double fps,
        int width,
        int height
    ) = 0;

    virtual void close() = 0;

    virtual bool write(
        const Frame& frame
    ) = 0;

    virtual bool isOpen() const noexcept = 0;

    virtual const std::string&
    lastError() const noexcept = 0;

protected:
    FrameSink() = default;
};

} // namespace video