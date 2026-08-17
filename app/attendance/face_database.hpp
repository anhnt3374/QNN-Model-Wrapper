#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace attendance {

constexpr std::size_t EMBEDDING_DIM =
    512;


struct FaceTemplate {
    std::string personId;

    std::string name;

    std::string phone;

    std::string pose;

    std::array<
        float,
        EMBEDDING_DIM
    > embedding{};
};


struct FaceMatch {
    bool matched =
        false;

    std::string personId;

    std::string name;

    std::string phone;

    std::string pose;

    float similarity =
        -1.0F;
};


class FaceDatabase {
public:
    bool load(
        const std::string& path
    );

    FaceMatch search(
        const std::array<
            float,
            EMBEDDING_DIM
        >& embedding,
        float threshold
    ) const;

    std::size_t templateCount() const noexcept;

    std::size_t personCount() const noexcept;

    bool empty() const noexcept;

    const std::string&
    lastError() const noexcept;

private:
    static bool readEmbedding(
        const cv::FileNode& node,
        std::array<
            float,
            EMBEDDING_DIM
        >& output
    );

    static bool normalize(
        std::array<
            float,
            EMBEDDING_DIM
        >& embedding
    );

private:
    std::vector<
        FaceTemplate
    > templates_;

    std::size_t personCount_ =
        0;

    std::string lastError_;
};

} // namespace attendance