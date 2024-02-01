cmake_minimum_required(VERSION 3.27)
project(vad LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(onnxruntime)
find_package(fmt)
add_executable(vad silero-vad-onnx.cpp)

target_include_directories(vad PRIVATE
        "${OpenCV_INCLUDE_DIRS}"
        "${spdlog_headers_directory}")

target_link_libraries(vad PRIVATE
        onnxruntime::onnxruntime
        fmt::fmt)
