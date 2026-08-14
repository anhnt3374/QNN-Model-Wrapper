#pragma once

#include <string>

namespace inference {

class SharedLibrary {
public:
    SharedLibrary() = default;

    explicit SharedLibrary(
        const std::string& path
    );

    ~SharedLibrary();

    SharedLibrary(
        const SharedLibrary&
    ) = delete;

    SharedLibrary& operator=(
        const SharedLibrary&
    ) = delete;

    SharedLibrary(
        SharedLibrary&& other
    ) noexcept;

    SharedLibrary& operator=(
        SharedLibrary&& other
    ) noexcept;

    bool open(const std::string& path);

    void close();

    bool isOpen() const noexcept;

    const std::string& path() const noexcept;

    const std::string& lastError() const noexcept;

    template<typename T>
    T getSymbol(const char* name)
    {
        return reinterpret_cast<T>(
            getSymbolRaw(name)
        );
    }

private:
    void* getSymbolRaw(const char* name);

private:
    void* handle_ = nullptr;

    std::string path_;

    std::string lastError_;
};

}
