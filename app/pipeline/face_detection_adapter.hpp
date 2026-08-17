#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace pipeline::face_adapter {

// ==========================================================
// Detection bbox layout traits
// ==========================================================

// x1, y1, x2, y2
template <
    typename T,
    typename = void
>
struct HasX1Y1X2Y2
    : std::false_type {
};

template <typename T>
struct HasX1Y1X2Y2<
    T,
    std::void_t<
        decltype(
            std::declval<T>().x1
        ),
        decltype(
            std::declval<T>().y1
        ),
        decltype(
            std::declval<T>().x2
        ),
        decltype(
            std::declval<T>().y2
        )
    >
> : std::true_type {
};


// x, y, w, h
template <
    typename T,
    typename = void
>
struct HasXYWH
    : std::false_type {
};

template <typename T>
struct HasXYWH<
    T,
    std::void_t<
        decltype(
            std::declval<T>().x
        ),
        decltype(
            std::declval<T>().y
        ),
        decltype(
            std::declval<T>().w
        ),
        decltype(
            std::declval<T>().h
        )
    >
> : std::true_type {
};


// x, y, width, height
template <
    typename T,
    typename = void
>
struct HasXYWidthHeight
    : std::false_type {
};

template <typename T>
struct HasXYWidthHeight<
    T,
    std::void_t<
        decltype(
            std::declval<T>().x
        ),
        decltype(
            std::declval<T>().y
        ),
        decltype(
            std::declval<T>().width
        ),
        decltype(
            std::declval<T>().height
        )
    >
> : std::true_type {
};


// bbox member exists
template <
    typename T,
    typename = void
>
struct HasBboxMember
    : std::false_type {
};

template <typename T>
struct HasBboxMember<
    T,
    std::void_t<
        decltype(
            std::declval<T>().bbox
        )
    >
> : std::true_type {
};


// box member exists
template <
    typename T,
    typename = void
>
struct HasBoxMember
    : std::false_type {
};

template <typename T>
struct HasBoxMember<
    T,
    std::void_t<
        decltype(
            std::declval<T>().box
        )
    >
> : std::true_type {
};


// ==========================================================
// Score traits
// ==========================================================

template <
    typename T,
    typename = void
>
struct HasScore
    : std::false_type {
};

template <typename T>
struct HasScore<
    T,
    std::void_t<
        decltype(
            std::declval<T>().score
        )
    >
> : std::true_type {
};


template <
    typename T,
    typename = void
>
struct HasConfidence
    : std::false_type {
};

template <typename T>
struct HasConfidence<
    T,
    std::void_t<
        decltype(
            std::declval<T>().confidence
        )
    >
> : std::true_type {
};


// ==========================================================
// Landmark traits
// ==========================================================

template <
    typename T,
    typename = void
>
struct HasLandmarks
    : std::false_type {
};

template <typename T>
struct HasLandmarks<
    T,
    std::void_t<
        decltype(
            std::declval<T>().landmarks
        )
    >
> : std::true_type {
};


template <
    typename T,
    typename = void
>
struct HasKeypoints
    : std::false_type {
};

template <typename T>
struct HasKeypoints<
    T,
    std::void_t<
        decltype(
            std::declval<T>().keypoints
        )
    >
> : std::true_type {
};


template <
    typename T,
    typename = void
>
struct HasKps
    : std::false_type {
};

template <typename T>
struct HasKps<
    T,
    std::void_t<
        decltype(
            std::declval<T>().kps
        )
    >
> : std::true_type {
};


// ==========================================================
// Container helpers
// ==========================================================

template <
    typename T,
    typename = void
>
struct IsIndexable
    : std::false_type {
};

template <typename T>
struct IsIndexable<
    T,
    std::void_t<
        decltype(
            std::declval<const T&>()[0]
        )
    >
> : std::true_type {
};


template <
    typename T,
    typename = void
>
struct ContainerValueType {
};

template <typename T>
struct ContainerValueType<
    T,
    std::void_t<
        typename T::value_type
    >
> {
    using type =
        typename T::value_type;
};


template <typename T>
using ContainerValueTypeT =
    typename ContainerValueType<T>::type;


// ==========================================================
// Generic bbox conversion
// ==========================================================

template <typename Box>
cv::Rect2f convertBox(
    const Box& box
)
{
    if constexpr (
        HasXYWH<Box>::value
    ) {

        return cv::Rect2f(
            static_cast<float>(
                box.x
            ),
            static_cast<float>(
                box.y
            ),
            static_cast<float>(
                box.w
            ),
            static_cast<float>(
                box.h
            )
        );
    }

    else if constexpr (
        HasXYWidthHeight<Box>::value
    ) {

        return cv::Rect2f(
            static_cast<float>(
                box.x
            ),
            static_cast<float>(
                box.y
            ),
            static_cast<float>(
                box.width
            ),
            static_cast<float>(
                box.height
            )
        );
    }

    else if constexpr (
        IsIndexable<Box>::value
    ) {

        // Assume:
        //
        // [x, y, width, height]
        //
        return cv::Rect2f(
            static_cast<float>(
                box[0]
            ),
            static_cast<float>(
                box[1]
            ),
            static_cast<float>(
                box[2]
            ),
            static_cast<float>(
                box[3]
            )
        );
    }

    else {

        static_assert(
            sizeof(Box) == 0,
            "Unsupported bbox representation"
        );
    }
}


// ==========================================================
// Detection → cv::Rect2f
// ==========================================================

template <typename Detection>
cv::Rect2f boundingBox(
    const Detection& detection
)
{
    // ------------------------------------------------------
    // Layout:
    //
    // float x;
    // float y;
    // float w;
    // float h;
    //
    // This is the layout used by the current SCRFD wrapper.
    // ------------------------------------------------------

    if constexpr (
        HasXYWH<Detection>::value
    ) {

        return cv::Rect2f(
            static_cast<float>(
                detection.x
            ),
            static_cast<float>(
                detection.y
            ),
            static_cast<float>(
                detection.w
            ),
            static_cast<float>(
                detection.h
            )
        );
    }

    // ------------------------------------------------------
    // x / y / width / height
    // ------------------------------------------------------

    else if constexpr (
        HasXYWidthHeight<Detection>::value
    ) {

        return cv::Rect2f(
            static_cast<float>(
                detection.x
            ),
            static_cast<float>(
                detection.y
            ),
            static_cast<float>(
                detection.width
            ),
            static_cast<float>(
                detection.height
            )
        );
    }

    // ------------------------------------------------------
    // x1 / y1 / x2 / y2
    // ------------------------------------------------------

    else if constexpr (
        HasX1Y1X2Y2<Detection>::value
    ) {

        return cv::Rect2f(
            static_cast<float>(
                detection.x1
            ),
            static_cast<float>(
                detection.y1
            ),
            static_cast<float>(
                detection.x2
                -
                detection.x1
            ),
            static_cast<float>(
                detection.y2
                -
                detection.y1
            )
        );
    }

    // ------------------------------------------------------
    // bbox member
    // ------------------------------------------------------

    else if constexpr (
        HasBboxMember<Detection>::value
    ) {

        return convertBox(
            detection.bbox
        );
    }

    // ------------------------------------------------------
    // box member
    // ------------------------------------------------------

    else if constexpr (
        HasBoxMember<Detection>::value
    ) {

        return convertBox(
            detection.box
        );
    }

    else {

        static_assert(
            sizeof(Detection) == 0,
            "Unsupported face bbox layout"
        );
    }
}


// ==========================================================
// Detection confidence
// ==========================================================

template <typename Detection>
float confidence(
    const Detection& detection
)
{
    if constexpr (
        HasScore<Detection>::value
    ) {

        return static_cast<float>(
            detection.score
        );
    }

    else if constexpr (
        HasConfidence<Detection>::value
    ) {

        return static_cast<float>(
            detection.confidence
        );
    }

    else {

        return 1.0F;
    }
}


// ==========================================================
// Landmark conversion
//
// Supports:
//
// 1.
//
// std::array<cv::Point2f, 5>
//
// 2.
//
// std::vector<cv::Point2f>
//
// 3.
//
// std::array<float, 10>
//
// layout:
//
// [
//     x0, y0,
//     x1, y1,
//     x2, y2,
//     x3, y3,
//     x4, y4
// ]
//
// Current SCRFD wrapper uses case #3.
// ==========================================================

template <typename PointContainer>
std::array<
    cv::Point2f,
    5
>
convertPoints(
    const PointContainer& points
)
{
    std::array<
        cv::Point2f,
        5
    > result{};

    using ValueType =
        ContainerValueTypeT<
            PointContainer
        >;

    // ------------------------------------------------------
    // Flat numeric array:
    //
    // [x0,y0,x1,y1,...]
    // ------------------------------------------------------

    if constexpr (
        std::is_arithmetic<
            ValueType
        >::value
    ) {

        for (std::size_t i = 0;
             i < 5;
             ++i) {

            result[i] =
                cv::Point2f(
                    static_cast<float>(
                        points[
                            i * 2
                        ]
                    ),
                    static_cast<float>(
                        points[
                            i * 2 + 1
                        ]
                    )
                );
        }
    }

    // ------------------------------------------------------
    // Point objects:
    //
    // points[i].x
    // points[i].y
    // ------------------------------------------------------

    else {

        for (std::size_t i = 0;
             i < 5;
             ++i) {

            result[i] =
                cv::Point2f(
                    static_cast<float>(
                        points[i].x
                    ),
                    static_cast<float>(
                        points[i].y
                    )
                );
        }
    }

    return result;
}


// ==========================================================
// Detection landmarks
// ==========================================================

template <typename Detection>
std::array<
    cv::Point2f,
    5
>
landmarks(
    const Detection& detection
)
{
    if constexpr (
        HasLandmarks<Detection>::value
    ) {

        return convertPoints(
            detection.landmarks
        );
    }

    else if constexpr (
        HasKeypoints<Detection>::value
    ) {

        return convertPoints(
            detection.keypoints
        );
    }

    else if constexpr (
        HasKps<Detection>::value
    ) {

        return convertPoints(
            detection.kps
        );
    }

    else {

        static_assert(
            sizeof(Detection) == 0,
            "Unsupported face landmark layout"
        );
    }
}

} // namespace pipeline::face_adapter