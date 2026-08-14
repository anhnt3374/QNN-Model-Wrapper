#include "inference/shared_library.hpp"

#include <dlfcn.h>

#include <utility>

namespace inference {

SharedLibrary::SharedLibrary(
    const std::string& path
)
{
    open(path);
}

SharedLibrary::~SharedLibrary()
{
    close();
}

SharedLibrary::SharedLibrary(
    SharedLibrary&& other
) noexcept
    : handle_(other.handle_),
      path_(std::move(other.path_)),
      lastError_(std::move(other.lastError_))
{
    other.handle_ = nullptr;
}

SharedLibrary& SharedLibrary::operator=(
    SharedLibrary&& other
) noexcept
{
    if (this == &other) {
        return *this;
    }

    close();

    handle_ = other.handle_;

    path_ = std::move(other.path_);

    lastError_ = std::move(other.lastError_);

    other.handle_ = nullptr;

    return *this;
}

bool SharedLibrary::open(
    const std::string& path
)
{
    close();

    dlerror();

    handle_ = dlopen(
        path.c_str(),
        RTLD_NOW | RTLD_LOCAL
    );

    if (handle_ == nullptr) {
        const char* error = dlerror();

        if (error != nullptr) {
            lastError_ = error;
        } else {
            lastError_ =
                "Unknown dlopen error";
        }

        return false;
    }

    path_ = path;

    lastError_.clear();

    return true;
}

void SharedLibrary::close()
{
    if (handle_ == nullptr) {
        return;
    }

    dlclose(handle_);

    handle_ = nullptr;

    path_.clear();
}

bool SharedLibrary::isOpen() const noexcept
{
    return handle_ != nullptr;
}

const std::string&
SharedLibrary::path() const noexcept
{
    return path_;
}

const std::string&
SharedLibrary::lastError() const noexcept
{
    return lastError_;
}

void* SharedLibrary::getSymbolRaw(
    const char* name
)
{
    if (handle_ == nullptr) {
        lastError_ =
            "Library is not open";

        return nullptr;
    }

    dlerror();

    void* symbol =
        dlsym(handle_, name);

    const char* error =
        dlerror();

    if (error != nullptr) {
        lastError_ = error;

        return nullptr;
    }

    lastError_.clear();

    return symbol;
}

}
