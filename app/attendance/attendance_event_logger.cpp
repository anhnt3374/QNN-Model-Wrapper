#include "attendance/attendance_event_logger.hpp"

#include <filesystem>
#include <iomanip>

namespace attendance {

AttendanceEventLogger::~AttendanceEventLogger()
{
    close();
}

bool AttendanceEventLogger::open(
    const std::string& path
)
{
    close();

    lastError_.clear();

    const std::filesystem::path filePath(
        path
    );

    std::error_code error;

    if (filePath.has_parent_path()) {

        std::filesystem::create_directories(
            filePath.parent_path(),
            error
        );

        if (error) {

            lastError_ =
                "Cannot create attendance log directory: "
                +
                error.message();

            return false;
        }
    }

    file_.open(
        path,
        std::ios::out
        |
        std::ios::trunc
    );

    if (!file_) {

        lastError_ =
            "Cannot open attendance event log: "
            +
            path;

        return false;
    }

    return true;
}

void AttendanceEventLogger::close()
{
    if (file_.is_open()) {

        file_.close();
    }
}

bool AttendanceEventLogger::write(
    const AttendanceEvent& event
)
{
    if (!file_.is_open()) {

        lastError_ =
            "Attendance event log is not open";

        return false;
    }

    file_
        << "{"
        << "\"frame_id\":"
        << event.frameId
        << ","
        << "\"video_timestamp_us\":"
        << event.videoTimestampUs
        << ","
        << "\"person_id\":\""
        << escapeJson(
            event.personId
        )
        << "\","
        << "\"name\":\""
        << escapeJson(
            event.name
        )
        << "\","
        << "\"phone\":\""
        << escapeJson(
            event.phone
        )
        << "\","
        << "\"similarity\":"
        << std::fixed
        << std::setprecision(6)
        << event.similarity
        << "}"
        << '\n';

    file_.flush();

    if (!file_) {

        lastError_ =
            "Cannot write attendance event";

        return false;
    }

    return true;
}

bool AttendanceEventLogger::isOpen() const noexcept
{
    return
        file_.is_open();
}

const std::string&
AttendanceEventLogger::lastError() const noexcept
{
    return
        lastError_;
}

std::string AttendanceEventLogger::escapeJson(
    const std::string& value
)
{
    std::string result;

    for (const char character :
         value) {

        switch (character) {

        case '\\':
            result +=
                "\\\\";
            break;

        case '"':
            result +=
                "\\\"";
            break;

        case '\n':
            result +=
                "\\n";
            break;

        case '\r':
            result +=
                "\\r";
            break;

        case '\t':
            result +=
                "\\t";
            break;

        default:
            result.push_back(
                character
            );
            break;
        }
    }

    return result;
}

} // namespace attendance