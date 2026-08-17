#pragma once

#include "video/frame_source.hpp"

#include <opencv2/videoio.hpp>

#include <cstdint>
#include <string>

namespace video {

class VideoFileSource final
    : public FrameSource {
public:
    explicit VideoFileSource(
        std::string path
    );

    ~VideoFileSource() override;

    bool open() override;

    void close() override;

    bool read(
        Frame& frame
    ) override;

    bool isOpen() const noexcept override;

    double fps() const noexcept override;

    int width() const noexcept override;

    int height() const noexcept override;

    uint64_t frameCount() const noexcept override;

    const std::string&
    lastError() const noexcept override;

    const std::string&
    path() const noexcept;

private:
    bool readMetadata();

private:
    std::string path_;

    cv::VideoCapture capture_;

    uint64_t nextFrameId_ = 0;

    double fps_ = 0.0;

    int width_ = 0;

    int height_ = 0;

    uint64_t frameCount_ = 0;

    std::string lastError_;
};

} // namespace video