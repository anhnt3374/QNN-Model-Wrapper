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
    imgproc
    imgcodecs
    videoio
    calib3d
)

find_package(
    Threads REQUIRED
)


# ==========================================================
# Existing inference module
# ==========================================================

if(NOT TARGET person_detection_model)

    add_subdirectory(
        ${CMAKE_CURRENT_SOURCE_DIR}/../inference
        ${CMAKE_BINARY_DIR}/inference
    )

endif()


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
# Benchmark report
# ==========================================================

add_library(
    benchmark_report
    STATIC

    benchmark/benchmark_report.cpp
)

target_include_directories(
    benchmark_report
    PUBLIC

    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    benchmark_report
    PUBLIC

    metrics_core
)


# ==========================================================
# Face tracking
# ==========================================================

add_library(
    face_tracking
    STATIC

    tracking/face_track_manager.cpp
)

target_include_directories(
    face_tracking
    PUBLIC

    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    face_tracking
    PUBLIC

    ${OpenCV_LIBS}
)


# ==========================================================
# Attendance core
# ==========================================================

add_library(
    attendance_core
    STATIC

    attendance/face_alignment.cpp
    attendance/face_database.cpp
    attendance/attendance_event_logger.cpp
)

target_include_directories(
    attendance_core
    PUBLIC

    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    attendance_core
    PUBLIC

    ${OpenCV_LIBS}
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
# Person pipeline
# ==========================================================

add_library(
    person_video_pipeline
    STATIC

    pipeline/person_pipeline.cpp
)

target_include_directories(
    person_video_pipeline
    PUBLIC

    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    person_video_pipeline
    PUBLIC

    video_core
    person_detection_model
    ${OpenCV_LIBS}
)


# ==========================================================
# FireSmoke pipeline
# ==========================================================

add_library(
    fire_smoke_video_pipeline
    STATIC

    pipeline/fire_smoke_pipeline.cpp
)

target_include_directories(
    fire_smoke_video_pipeline
    PUBLIC

    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    fire_smoke_video_pipeline
    PUBLIC

    video_core
    fire_smoke_detection_model
    ${OpenCV_LIBS}
)


# ==========================================================
# Attendance pipeline
# ==========================================================

add_library(
    attendance_video_pipeline
    STATIC

    pipeline/attendance_pipeline.cpp
)

target_include_directories(
    attendance_video_pipeline
    PUBLIC

    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    attendance_video_pipeline
    PUBLIC

    video_core

    attendance_core
    face_tracking

    face_detection_model
    face_embedding_model

    ${OpenCV_LIBS}
)


# ==========================================================
# Composite
# ==========================================================

add_library(
    composite_frame_processor
    STATIC

    pipeline/composite_frame_processor.cpp
)

target_include_directories(
    composite_frame_processor
    PUBLIC

    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    composite_frame_processor
    PUBLIC

    video_core
)


# ==========================================================
# Normal V6 application
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
    attendance_core
    face_tracking

    composite_frame_processor

    person_video_pipeline
    fire_smoke_video_pipeline
    attendance_video_pipeline

    person_detection_model
    fire_smoke_detection_model
    face_detection_model
    face_embedding_model
)


# ==========================================================
# V7 benchmark executable
# ==========================================================

add_executable(
    qcs6490_benchmark_app

    benchmark_main.cpp
)

target_link_libraries(
    qcs6490_benchmark_app
    PRIVATE

    benchmark_report

    video_core
    metrics_core
    attendance_core
    face_tracking

    composite_frame_processor

    person_video_pipeline
    fire_smoke_video_pipeline
    attendance_video_pipeline

    person_detection_model
    fire_smoke_detection_model
    face_detection_model
    face_embedding_model
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


# ==========================================================
# V6 tracker test
# ==========================================================

add_executable(
    test_face_track_manager

    tests/test_face_track_manager.cpp
)

target_link_libraries(
    test_face_track_manager
    PRIVATE

    face_tracking
)


# ==========================================================
# V7 benchmark report test
# ==========================================================

add_executable(
    test_benchmark_report

    tests/test_benchmark_report.cpp
)

target_link_libraries(
    test_benchmark_report
    PRIVATE

    benchmark_report
    metrics_core
)

# ==========================================================
# V8 async runtime
# ==========================================================

add_library(
    async_runtime
    STATIC

    async_runtime/async_metrics_logger.cpp
    async_runtime/branch_worker.cpp
    async_runtime/async_video_pipeline.cpp
)

target_include_directories(
    async_runtime
    PUBLIC

    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    async_runtime
    PUBLIC

    video_core
    metrics_core

    ${OpenCV_LIBS}

    Threads::Threads
)


# ==========================================================
# V8 async application
# ==========================================================

add_executable(
    qcs6490_async_app

    async_main.cpp
)

target_link_libraries(
    qcs6490_async_app
    PRIVATE

    async_runtime

    video_core
    metrics_core

    attendance_core
    face_tracking

    person_video_pipeline
    fire_smoke_video_pipeline
    attendance_video_pipeline

    person_detection_model
    fire_smoke_detection_model
    face_detection_model
    face_embedding_model

    ${OpenCV_LIBS}

    Threads::Threads
)


# ==========================================================
# V8 primitive tests
# ==========================================================

add_executable(
    test_latest_frame_queue

    tests/test_latest_frame_queue.cpp
)

target_include_directories(
    test_latest_frame_queue
    PRIVATE

    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    test_latest_frame_queue
    PRIVATE

    Threads::Threads
)


add_executable(
    test_htp_execution_gate

    tests/test_htp_execution_gate.cpp
)

target_include_directories(
    test_htp_execution_gate
    PRIVATE

    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    test_htp_execution_gate
    PRIVATE

    Threads::Threads
)