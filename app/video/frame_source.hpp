#pragma once

#include "video/frame.hpp"

#include <string>

namespace video {

class FrameSource {
public:
    virtual ~FrameSource() = default;

    FrameSource(
        const FrameSource&
    ) = delete;

    FrameSource& operator=(
        const FrameSource&
    ) = delete;

    virtual bool open() = 0;

    virtual void close() = 0;

    virtual bool read(
        Frame& frame
    ) = 0;

    virtual bool isOpen() const noexcept = 0;

    virtual double fps() const noexcept = 0;

    virtual int width() const noexcept = 0;

    virtual int height() const noexcept = 0;

    virtual uint64_t frameCount() const noexcept = 0;

    virtual const std::string&
    lastError() const noexcept = 0;

protected:
    FrameSource() = default;
};

} // namespace video