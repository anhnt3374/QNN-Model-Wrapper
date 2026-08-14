#pragma once

#include "inference/shared_library.hpp"

#include <QnnInterface.h>

#include <cstdint>
#include <string>

namespace inference {

class QnnBackend {
public:
    bool loadLibrary(
        const std::string& backendPath
    );

    bool loadProviders();

    uint32_t providerCount() const noexcept;

    const std::string& lastError() const noexcept;

private:
    SharedLibrary backendLibrary_;

    const QnnInterface_t** providers_ = nullptr;

    uint32_t providerCount_ = 0;

    std::string lastError_;
};

}