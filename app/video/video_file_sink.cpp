#include "video/video_file_sink.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <numeric>
#include <sstream>
#include <utility>

namespace video {

VideoFileSink::VideoFileSink(
    std::string path,
    int bitrateKbps
)
    : path_(
          std::move(
              path
          )
      ),
      bitrateKbps_(
          bitrateKbps
      )
{
}

VideoFileSink::~VideoFileSink()
{
    close();
}

bool VideoFileSink::buildPipeline(
    double fps,
    int width,
    int height
)
{
    if (!std::isfinite(fps) ||
        fps <= 0.0) {

        lastError_ =
            "Invalid output FPS";

        return false;
    }

    // Convert FPS to a GStreamer fraction.
    //
    // Examples:
    //
    // 30.0   -> 30/1
    // 25.0   -> 25/1
    // 29.97  -> 2997/100
    //
    constexpr int DENOMINATOR_BASE =
        1000;

    int numerator =
        static_cast<int>(
            std::lround(
                fps
                *
                DENOMINATOR_BASE
            )
        );

    int denominator =
        DENOMINATOR_BASE;

    const int divisor =
        std::gcd(
            numerator,
            denominator
        );

    numerator /=
        divisor;

    denominator /=
        divisor;

    const std::string escapedPath =
        escapeGstreamerString(
            path_
        );

    std::ostringstream oss;

    // OpenCV pushes CV_8UC3 BGR frames into appsrc.
    //
    // We intentionally DO NOT force:
    //
    //   output-io-mode=5
    //   capture-io-mode=4
    //
    // because cv::Mat/appsrc buffers are system-memory.
    //
    // QCS6490 v4l2h264enc successfully handles this path
    // using its default I/O mode.

    oss
        << "appsrc "
        << "! videoconvert "
        << "! video/x-raw,"
        << "format=NV12,"
        << "width="
        << width
        << ",height="
        << height
        << ",framerate="
        << numerator
        << "/"
        << denominator
        << " "
        << "! v4l2h264enc "
        << "! h264parse "
        << "! video/x-h264,"
        << "stream-format=avc,"
        << "alignment=au "
        << "! mp4mux "
        << "! filesink location=\""
        << escapedPath
        << "\" sync=false";

    pipeline_ =
        oss.str();

    return true;
}

bool VideoFileSink::open(
    double fps,
    int width,
    int height
)
{
    close();

    lastError_.clear();

    if (path_.empty()) {

        lastError_ =
            "Video output path is empty";

        return false;
    }

    if (!std::isfinite(
            fps
        ) ||
        fps <= 0.0) {

        lastError_ =
            "Invalid output FPS";

        return false;
    }

    if (width <= 0 ||
        height <= 0) {

        lastError_ =
            "Invalid output video dimensions";

        return false;
    }

    if (bitrateKbps_ <= 0) {

        lastError_ =
            "Invalid output bitrate";

        return false;
    }

    if (!buildPipeline(
            fps,
            width,
            height
        )) {

        return false;
    }

    // =====================================================
    // Important:
    //
    // CAP_GSTREAMER
    //
    // fourcc = 0 because encoding is explicitly described
    // inside the GStreamer pipeline.
    // =====================================================

    if (!writer_.open(
            pipeline_,
            cv::CAP_GSTREAMER,
            0,
            fps,
            cv::Size(
                width,
                height
            ),
            true
        )) {

        lastError_ =
            "Cannot open QCS6490 H.264 output pipeline:\n"
            +
            pipeline_;

        return false;
    }

    fps_ =
        fps;

    width_ =
        width;

    height_ =
        height;

    writtenFrames_ =
        0;

    lastError_.clear();

    return true;
}

void VideoFileSink::close()
{
    if (writer_.isOpened()) {

        writer_.release();
    }

    fps_ =
        0.0;

    width_ =
        0;

    height_ =
        0;
}

bool VideoFileSink::write(
    const Frame& frame
)
{
    if (!writer_.isOpened()) {

        lastError_ =
            "VideoFileSink is not open";

        return false;
    }

    if (frame.image.empty()) {

        lastError_ =
            "Cannot write an empty frame";

        return false;
    }

    if (frame.image.cols !=
            width_ ||
        frame.image.rows !=
            height_) {

        std::ostringstream oss;

        oss
            << "Output frame size mismatch: got "
            << frame.image.cols
            << "x"
            << frame.image.rows
            << ", expected "
            << width_
            << "x"
            << height_;

        lastError_ =
            oss.str();

        return false;
    }

    if (frame.image.type() !=
        CV_8UC3) {

        lastError_ =
            "VideoFileSink expects CV_8UC3 BGR frames";

        return false;
    }

    writer_.write(
        frame.image
    );

    ++writtenFrames_;

    lastError_.clear();

    return true;
}

bool VideoFileSink::isOpen() const noexcept
{
    return
        writer_.isOpened();
}

const std::string&
VideoFileSink::lastError() const noexcept
{
    return
        lastError_;
}

uint64_t
VideoFileSink::writtenFrames() const noexcept
{
    return
        writtenFrames_;
}

const std::string&
VideoFileSink::path() const noexcept
{
    return
        path_;
}

const std::string&
VideoFileSink::pipeline() const noexcept
{
    return
        pipeline_;
}

std::string
VideoFileSink::escapeGstreamerString(
    const std::string& value
)
{
    std::string result;

    result.reserve(
        value.size()
    );

    for (const char ch :
         value) {

        if (ch == '\\' ||
            ch == '"') {

            result.push_back(
                '\\'
            );
        }

        result.push_back(
            ch
        );
    }

    return
        result;
}

} // namespace video