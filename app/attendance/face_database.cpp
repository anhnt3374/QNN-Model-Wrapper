#include "attendance/face_database.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <set>
#include <utility>

namespace attendance {

bool FaceDatabase::load(
    const std::string& path
)
{
    templates_.clear();

    personCount_ =
        0;

    lastError_.clear();

    // =====================================================
    // Validate path
    // =====================================================

    if (path.empty()) {

        lastError_ =
            "Face database path is empty";

        return false;
    }

    namespace fs =
        std::filesystem;

    std::error_code error;

    if (!fs::exists(
            path,
            error
        )) {

        lastError_ =
            "Face database does not exist: "
            +
            path;

        return false;
    }

    if (error) {

        lastError_ =
            "Cannot inspect face database: "
            +
            error.message();

        return false;
    }

    if (!fs::is_regular_file(
            path,
            error
        )) {

        lastError_ =
            "Face database is not a regular file: "
            +
            path;

        return false;
    }

    if (error) {

        lastError_ =
            "Cannot inspect face database: "
            +
            error.message();

        return false;
    }

    // =====================================================
    // IMPORTANT:
    //
    // Empty database is valid in V5.
    //
    // This lets us run:
    //
    // SCRFD
    // → alignment
    // → EdgeFace
    // → search(empty DB)
    // → UNKNOWN
    //
    // without requiring enrollment data.
    //
    // We must detect this BEFORE cv::FileStorage because
    // OpenCV may throw on a zero-byte JSON file.
    // =====================================================

    const std::uintmax_t fileSize =
        fs::file_size(
            path,
            error
        );

    if (error) {

        lastError_ =
            "Cannot get face database size: "
            +
            error.message();

        return false;
    }

    if (fileSize == 0) {

        // Valid empty database.
        //
        // templates_ stays empty.
        // personCount_ stays 0.

        return true;
    }

    // =====================================================
    // Open JSON
    // =====================================================

    cv::FileStorage storage;

    try {

        const bool opened =
            storage.open(
                path,
                cv::FileStorage::READ
                |
                cv::FileStorage::FORMAT_JSON
            );

        if (!opened) {

            lastError_ =
                "Cannot open face database: "
                +
                path;

            return false;
        }
    }
    catch (const cv::Exception& exception) {

        lastError_ =
            "OpenCV cannot parse face database '"
            +
            path
            +
            "': "
            +
            exception.what();

        return false;
    }

    // =====================================================
    // Expected formats:
    //
    // {
    //   "persons": [...]
    // }
    //
    // or:
    //
    // [
    //   {...},
    //   {...}
    // ]
    // =====================================================

    cv::FileNode persons =
        storage["persons"];

    if (persons.empty()) {

        const cv::FileNode root =
            storage.root();

        if (root.isSeq()) {

            persons =
                root;
        }
    }

    // =====================================================
    // Valid JSON but no persons.
    //
    // Example:
    //
    // {
    //   "persons": []
    // }
    //
    // Treat as a valid empty database.
    // =====================================================

    if (persons.empty()) {

        return true;
    }

    if (!persons.isSeq()) {

        lastError_ =
            "Face database 'persons' must be an array";

        return false;
    }

    // =====================================================
    // Parse people/templates
    // =====================================================

    std::set<std::string> uniquePersons;

    static const std::array<
        const char*,
        3
    > POSES = {
        "front",
        "left30",
        "right30"
    };

    for (auto iterator =
             persons.begin();
         iterator !=
             persons.end();
         ++iterator) {

        const cv::FileNode person =
            *iterator;

        std::string personId;
        std::string name;
        std::string phone;

        person["id"] >>
            personId;

        person["name"] >>
            name;

        person["phone"] >>
            phone;

        if (personId.empty()) {

            personId =
                phone;
        }

        if (personId.empty()) {

            personId =
                name;
        }

        if (name.empty()) {

            continue;
        }

        cv::FileNode embeddings =
            person["embeddings"];

        // Support legacy structure where front/left30/right30
        // may be directly inside each person object.
        if (embeddings.empty()) {

            embeddings =
                person;
        }

        bool loadedPerson =
            false;

        for (const char* pose :
             POSES) {

            const cv::FileNode node =
                embeddings[pose];

            if (node.empty()) {

                continue;
            }

            FaceTemplate faceTemplate;

            faceTemplate.personId =
                personId;

            faceTemplate.name =
                name;

            faceTemplate.phone =
                phone;

            faceTemplate.pose =
                pose;

            if (!readEmbedding(
                    node,
                    faceTemplate.embedding
                )) {

                continue;
            }

            // Missing enrollment poses may be represented
            // by all-zero embeddings.
            //
            // Ignore those templates.
            if (!normalize(
                    faceTemplate.embedding
                )) {

                continue;
            }

            templates_.push_back(
                std::move(
                    faceTemplate
                )
            );

            loadedPerson =
                true;
        }

        if (loadedPerson) {

            uniquePersons.insert(
                personId
            );
        }
    }

    personCount_ =
        uniquePersons.size();

    // =====================================================
    // V5 behavior:
    //
    // ZERO usable templates is NOT an error.
    //
    // Database remains valid but every search will return
    // UNKNOWN.
    // =====================================================

    return true;
}


bool FaceDatabase::readEmbedding(
    const cv::FileNode& node,
    std::array<
        float,
        EMBEDDING_DIM
    >& output
)
{
    if (!node.isSeq() ||
        node.size() !=
            EMBEDDING_DIM) {

        return false;
    }

    std::size_t index =
        0;

    for (auto iterator =
             node.begin();
         iterator !=
             node.end();
         ++iterator) {

        if (index >=
            EMBEDDING_DIM) {

            return false;
        }

        output[index++] =
            static_cast<float>(
                static_cast<double>(
                    *iterator
                )
            );
    }

    return
        index ==
        EMBEDDING_DIM;
}


bool FaceDatabase::normalize(
    std::array<
        float,
        EMBEDDING_DIM
    >& embedding
)
{
    double squaredNorm =
        0.0;

    for (const float value :
         embedding) {

        if (!std::isfinite(
                value
            )) {

            return false;
        }

        squaredNorm +=
            static_cast<double>(
                value
            )
            *
            static_cast<double>(
                value
            );
    }

    const double norm =
        std::sqrt(
            squaredNorm
        );

    if (norm <
        1e-12) {

        return false;
    }

    for (float& value :
         embedding) {

        value =
            static_cast<float>(
                static_cast<double>(
                    value
                )
                /
                norm
            );
    }

    return true;
}


FaceMatch FaceDatabase::search(
    const std::array<
        float,
        EMBEDDING_DIM
    >& embedding,
    float threshold
) const
{
    FaceMatch best;

    // =====================================================
    // EMPTY DATABASE
    //
    // best defaults to:
    //
    // matched    = false
    // similarity = -1
    //
    // Therefore caller will render:
    //
    // unknown
    //
    // No special error required.
    // =====================================================

    if (templates_.empty()) {

        return best;
    }

    for (const auto& candidate :
         templates_) {

        double similarity =
            0.0;

        for (std::size_t i = 0;
             i < EMBEDDING_DIM;
             ++i) {

            similarity +=
                static_cast<double>(
                    embedding[i]
                )
                *
                static_cast<double>(
                    candidate.embedding[i]
                );
        }

        if (similarity >
            best.similarity) {

            best.similarity =
                static_cast<float>(
                    similarity
                );

            best.personId =
                candidate.personId;

            best.name =
                candidate.name;

            best.phone =
                candidate.phone;

            best.pose =
                candidate.pose;
        }
    }

    best.matched =
        best.similarity >=
        threshold;

    return best;
}


std::size_t
FaceDatabase::templateCount() const noexcept
{
    return
        templates_.size();
}


std::size_t
FaceDatabase::personCount() const noexcept
{
    return
        personCount_;
}


bool FaceDatabase::empty() const noexcept
{
    return
        templates_.empty();
}


const std::string&
FaceDatabase::lastError() const noexcept
{
    return
        lastError_;
}

} // namespace attendance