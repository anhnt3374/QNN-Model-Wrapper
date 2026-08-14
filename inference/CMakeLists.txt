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
# Inference runtime
# ==========================================================

add_library(
    inference_runtime
    STATIC

    src/shared_library.cpp
    src/qnn_backend.cpp
    src/qnn_context.cpp
    src/qnn_model.cpp
    src/qnn_tensor_buffer.cpp
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
# QNN backend/context test
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
# QNN model test
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