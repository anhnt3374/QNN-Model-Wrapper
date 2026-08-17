#pragma once

#include "video/frame_sink.hpp"

#include <opencv2/videoio.hpp>

#include <cstdint>
#include <string>

namespace video {

class VideoFileSink final
    : public FrameSink {
public:
    explicit VideoFileSink(
        std::string path,
        int bitrateKbps = 4000
    );

    ~VideoFileSink() override;

    bool open(
        double fps,
        int width,
        int height
    ) override;

    void close() override;

    bool write(
        const Frame& frame
    ) override;

    bool isOpen() const noexcept override;

    const std::string&
    lastError() const noexcept override;

    uint64_t writtenFrames() const noexcept;

    const std::string&
    path() const noexcept;

    const std::string&
    pipeline() const noexcept;

private:
    bool buildPipeline(
        double fps,
        int width,
        int height
    );

    static std::string escapeGstreamerString(
        const std::string& value
    );

private:
    std::string path_;

    int bitrateKbps_ = 4000;

    cv::VideoWriter writer_;

    double fps_ = 0.0;

    int width_ = 0;

    int height_ = 0;

    uint64_t writtenFrames_ = 0;

    std::string pipeline_;

    std::string lastError_;
};

} // namespace video