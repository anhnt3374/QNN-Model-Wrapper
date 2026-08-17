#pragma once

#include "models/fire_smoke_detection_model.hpp"

#include <array>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace config {

inline bool parseFireSmokeAnchorList(
    const std::string& text,
    std::array<
        models::FireSmokeAnchor,
        3
    >& anchors,
    std::string& error
)
{
    error.clear();

    std::vector<float> values;

    std::stringstream stream(
        text
    );

    std::string token;

    while (std::getline(
        stream,
        token,
        ','
    )) {

        try {

            std::size_t consumed =
                0;

            const float value =
                std::stof(
                    token,
                    &consumed
                );

            if (consumed !=
                token.size()) {

                error =
                    "Invalid anchor value: "
                    +
                    token;

                return false;
            }

            if (value <= 0.0F) {

                error =
                    "Anchor width/height must be > 0";

                return false;
            }

            values.push_back(
                value
            );
        }
        catch (...) {

            error =
                "Cannot parse anchor value: "
                +
                token;

            return false;
        }
    }

    if (values.size() != 6) {

        error =
            "Anchor list must contain exactly "
            "6 values: w1,h1,w2,h2,w3,h3";

        return false;
    }

    for (std::size_t i = 0;
         i < 3;
         ++i) {

        anchors[i].width =
            values[i * 2];

        anchors[i].height =
            values[i * 2 + 1];
    }

    return true;
}

inline bool loadFireSmokeAnchorsFromEnvironment(
    models::FireSmokeAnchors& anchors,
    std::string& error
)
{
    error.clear();

    const char* s4 =
        std::getenv(
            "QNN_FIRE_SMOKE_ANCHORS_S4"
        );

    const char* s8 =
        std::getenv(
            "QNN_FIRE_SMOKE_ANCHORS_S8"
        );

    const char* s16 =
        std::getenv(
            "QNN_FIRE_SMOKE_ANCHORS_S16"
        );

    if (s4 == nullptr) {

        error =
            "QNN_FIRE_SMOKE_ANCHORS_S4 is not set";

        return false;
    }

    if (s8 == nullptr) {

        error =
            "QNN_FIRE_SMOKE_ANCHORS_S8 is not set";

        return false;
    }

    if (s16 == nullptr) {

        error =
            "QNN_FIRE_SMOKE_ANCHORS_S16 is not set";

        return false;
    }

    if (!parseFireSmokeAnchorList(
            s4,
            anchors.s4,
            error
        )) {

        error =
            "Invalid S4 anchors: "
            +
            error;

        return false;
    }

    if (!parseFireSmokeAnchorList(
            s8,
            anchors.s8,
            error
        )) {

        error =
            "Invalid S8 anchors: "
            +
            error;

        return false;
    }

    if (!parseFireSmokeAnchorList(
            s16,
            anchors.s16,
            error
        )) {

        error =
            "Invalid S16 anchors: "
            +
            error;

        return false;
    }

    return true;
}

} // namespace config