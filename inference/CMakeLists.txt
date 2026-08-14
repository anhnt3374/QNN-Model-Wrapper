cmake_minimum_required(VERSION 3.16)

project(inference_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# ==========================================================
# QNN SDK
# ==========================================================

set(
    QNN_SDK_ROOT
    ""
    CACHE PATH
    "Path to Qualcomm QAIRT/QNN SDK"
)

if(NOT QNN_SDK_ROOT)
    message(
        FATAL_ERROR
        "QNN_SDK_ROOT must be provided"
    )
endif()


# ==========================================================
# OpenCV
#
# Generic inference_runtime does NOT depend on OpenCV.
# Only the model-specific FaceDetectionModel does.
# ==========================================================

find_package(
    OpenCV REQUIRED
    COMPONENTS
    core
    imgproc
)


# ==========================================================
# Generic inference runtime
# ==========================================================

add_library(
    inference_runtime
    STATIC

    src/shared_library.cpp
    src/qnn_backend.cpp
    src/qnn_context.cpp
    src/qnn_model.cpp
    src/qnn_tensor_buffer.cpp
    src/qnn_quantization.cpp
)

target_include_directories(
    inference_runtime
    PUBLIC

    ${CMAKE_CURRENT_SOURCE_DIR}/include

    ${QNN_SDK_ROOT}/include/QNN

    ${QNN_SDK_ROOT}/examples/QNN/SampleApp/SampleApp/src/WrapperUtils
)

target_link_libraries(
    inference_runtime
    PUBLIC

    dl
)


# ==========================================================
# SCRFD FaceDetectionModel
# ==========================================================

add_library(
    face_detection_model
    STATIC

    src/models/face_detection_model.cpp
)

target_include_directories(
    face_detection_model
    PUBLIC

    ${CMAKE_CURRENT_SOURCE_DIR}/include

    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    face_detection_model
    PUBLIC

    inference_runtime

    ${OpenCV_LIBS}
)


# ==========================================================
# SharedLibrary test plugin
# ==========================================================

add_library(
    test_plugin
    SHARED

    tests/test_plugin.cpp
)


# ==========================================================
# SharedLibrary test
# ==========================================================

add_executable(
    test_shared_library

    tests/test_shared_library.cpp
)

target_link_libraries(
    test_shared_library
    PRIVATE

    inference_runtime
)

target_compile_definitions(
    test_shared_library
    PRIVATE

    TEST_PLUGIN_PATH="$<TARGET_FILE:test_plugin>"
)


# ==========================================================
# QNN backend test
# ==========================================================

add_executable(
    test_qnn_backend

    tests/test_qnn_backend.cpp
)

target_link_libraries(
    test_qnn_backend
    PRIVATE

    inference_runtime
)


# ==========================================================
# Generic QNN model test
# ==========================================================

add_executable(
    test_qnn_model

    tests/test_qnn_model.cpp
)

target_link_libraries(
    test_qnn_model
    PRIVATE

    inference_runtime
)


# ==========================================================
# Quantization test
# ==========================================================

add_executable(
    test_qnn_quantization

    tests/test_qnn_quantization.cpp
)

target_link_libraries(
    test_qnn_quantization
    PRIVATE

    inference_runtime
)


# ==========================================================
# FaceDetectionModel preprocess test
# ==========================================================

add_executable(
    test_face_detection_model

    tests/test_face_detection_model.cpp
)

target_link_libraries(
    test_face_detection_model
    PRIVATE

    face_detection_model
)