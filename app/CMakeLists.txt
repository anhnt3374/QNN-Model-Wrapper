cmake_minimum_required(VERSION 3.16)

project(
    qcs6490_video_app
    LANGUAGES CXX
)

set(
    CMAKE_CXX_STANDARD
    17
)

set(
    CMAKE_CXX_STANDARD_REQUIRED
    ON
)

set(
    CMAKE_CXX_EXTENSIONS
    OFF
)


# ==========================================================
# OpenCV
# ==========================================================

find_package(
    OpenCV REQUIRED
    COMPONENTS

    core
    videoio
)


# ==========================================================
# Metrics
# ==========================================================

add_library(
    metrics_core
    STATIC

    metrics/profiler.cpp
    metrics/metrics_logger.cpp
)

target_include_directories(
    metrics_core
    PUBLIC

    ${CMAKE_CURRENT_SOURCE_DIR}
)


# ==========================================================
# Video core
# ==========================================================

add_library(
    video_core
    STATIC

    video/video_file_source.cpp
    video/video_file_sink.cpp

    pipeline/video_pipeline.cpp
)

target_include_directories(
    video_core
    PUBLIC

    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    video_core
    PUBLIC

    metrics_core
    ${OpenCV_LIBS}
)


# ==========================================================
# Main application
# ==========================================================

add_executable(
    qcs6490_video_app

    main.cpp
)

target_link_libraries(
    qcs6490_video_app
    PRIVATE

    video_core
    metrics_core
)


# ==========================================================
# V2 test
# ==========================================================

add_executable(
    test_video_pipeline

    tests/test_video_pipeline.cpp
)

target_link_libraries(
    test_video_pipeline
    PRIVATE

    video_core
    metrics_core
)