#pragma once

#include <cstdint>
#include <fstream>
#include <string>

namespace attendance {

struct AttendanceEvent {
    uint64_t frameId =
        0;

    int64_t videoTimestampUs =
        0;

    std::string personId;

    std::string name;

    std::string phone;

    float similarity =
        0.0F;
};


class AttendanceEventLogger {
public:
    AttendanceEventLogger() = default;

    ~AttendanceEventLogger();

    bool open(
        const std::string& path
    );

    void close();

    bool write(
        const AttendanceEvent& event
    );

    bool isOpen() const noexcept;

    const std::string&
    lastError() const noexcept;

private:
    static std::string escapeJson(
        const std::string& value
    );

private:
    std::ofstream file_;

    std::string lastError_;
};

} // namespace attendance