#include "video/video_file_source.hpp"

#include <cmath>
#include <cstdint>
#include <utility>

namespace video {

VideoFileSource::VideoFileSource(
    std::string path
)
    : path_(
          std::move(
              path
          )
      )
{
}

VideoFileSource::~VideoFileSource()
{
    close();
}

bool VideoFileSource::open()
{
    close();

    lastError_.clear();

    if (path_.empty()) {

        lastError_ =
            "Video input path is empty";

        return false;
    }

    if (!capture_.open(
            path_,
            cv::CAP_ANY
        )) {

        lastError_ =
            "Cannot open video file: "
            +
            path_;

        return false;
    }

    if (!readMetadata()) {

        capture_.release();

        return false;
    }

    nextFrameId_ =
        0;

    return true;
}

bool VideoFileSource::readMetadata()
{
    fps_ =
        capture_.get(
            cv::CAP_PROP_FPS
        );

    width_ =
        static_cast<int>(
            std::lround(
                capture_.get(
                    cv::CAP_PROP_FRAME_WIDTH
                )
            )
        );

    height_ =
        static_cast<int>(
            std::lround(
                capture_.get(
                    cv::CAP_PROP_FRAME_HEIGHT
                )
            )
        );

    const double rawFrameCount =
        capture_.get(
            cv::CAP_PROP_FRAME_COUNT
        );

    if (!std::isfinite(
            fps_
        ) ||
        fps_ <= 0.0) {

        lastError_ =
            "Invalid video FPS";

        return false;
    }

    if (width_ <= 0 ||
        height_ <= 0) {

        lastError_ =
            "Invalid video dimensions";

        return false;
    }

    if (std::isfinite(
            rawFrameCount
        ) &&
        rawFrameCount > 0.0) {

        frameCount_ =
            static_cast<uint64_t>(
                std::llround(
                    rawFrameCount
                )
            );
    }
    else {

        // Some video backends/streams do not know the
        // complete frame count in advance.
        frameCount_ =
            0;
    }

    return true;
}

void VideoFileSource::close()
{
    if (capture_.isOpened()) {

        capture_.release();
    }

    nextFrameId_ =
        0;

    fps_ =
        0.0;

    width_ =
        0;

    height_ =
        0;

    frameCount_ =
        0;
}

bool VideoFileSource::read(
    Frame& frame
)
{
    frame = {};

    if (!capture_.isOpened()) {

        lastError_ =
            "VideoFileSource is not open";

        return false;
    }

    cv::Mat image;

    if (!capture_.read(
            image
        )) {

        // End-of-file is intentionally represented by
        // read() == false.
        //
        // The pipeline can distinguish a normal EOF from
        // an actual source error using lastError().
        lastError_.clear();

        return false;
    }

    if (image.empty()) {

        lastError_ =
            "Video backend returned an empty frame";

        return false;
    }

    frame.frameId =
        nextFrameId_;

    // Use source timeline rather than wall-clock time.
    //
    // This makes offline file processing deterministic.
    const double timestampSeconds =
        static_cast<double>(
            nextFrameId_
        )
        /
        fps_;

    frame.captureTimestampUs =
        static_cast<int64_t>(
            std::llround(
                timestampSeconds
                *
                1'000'000.0
            )
        );

    frame.image =
        std::move(
            image
        );

    ++nextFrameId_;

    lastError_.clear();

    return true;
}

bool VideoFileSource::isOpen() const noexcept
{
    return
        capture_.isOpened();
}

double VideoFileSource::fps() const noexcept
{
    return
        fps_;
}

int VideoFileSource::width() const noexcept
{
    return
        width_;
}

int VideoFileSource::height() const noexcept
{
    return
        height_;
}

uint64_t
VideoFileSource::frameCount() const noexcept
{
    return
        frameCount_;
}

const std::string&
VideoFileSource::lastError() const noexcept
{
    return
        lastError_;
}

const std::string&
VideoFileSource::path() const noexcept
{
    return
        path_;
}

} // namespace video