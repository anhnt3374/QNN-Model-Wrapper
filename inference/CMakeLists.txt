cmake_minimum_required(VERSION 3.16)

project(inference_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(
    QNN_SDK_ROOT
    ""
    CACHE PATH
    "Path to Qualcomm QNN/QAIRT SDK"
)

if(NOT QNN_SDK_ROOT)
    message(
        FATAL_ERROR
        "QNN_SDK_ROOT must be provided"
    )
endif()


# =========================
# Runtime library
# =========================

add_library(
    inference_runtime
    STATIC

    src/shared_library.cpp
    src/qnn_backend.cpp
)

target_include_directories(
    inference_runtime
    PUBLIC

    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${QNN_SDK_ROOT}/include/QNN
)

target_link_libraries(
    inference_runtime
    PUBLIC

    dl
)


# =========================
# Test plugin
# =========================

add_library(
    test_plugin
    SHARED

    tests/test_plugin.cpp
)


# =========================
# SharedLibrary test
# =========================

add_executable(
    test_shared_library

    tests/test_shared_library.cpp
)

target_compile_definitions(
    test_shared_library
    PRIVATE
    TEST_PLUGIN_PATH="$<TARGET_FILE:test_plugin>"
)

target_link_libraries(
    test_shared_library
    PRIVATE

    inference_runtime
)

add_executable(
    test_qnn_backend
    tests/test_qnn_backend.cpp
)

target_link_libraries(
    test_qnn_backend
    PRIVATE
    inference_runtime
)
