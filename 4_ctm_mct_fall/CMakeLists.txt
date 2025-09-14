cmake_minimum_required(VERSION 3.10)
project(MCT_SEM_6)

set(CMAKE_CXX_STANDARD             20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_EXPORT_COMPILE_COMMANDS  ON)

set(Clang_DIR "/usr/lib/llvm-16/lib/cmake/clang")
set(LLVM_DIR "/usr/lib/llvm-16/lib/cmake/llvm")
find_package(Clang CONFIG)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak")

include_directories(${CMAKE_SOURCE_DIR}/external/INMOST_Practicum_2024)
include_directories(${CMAKE_SOURCE_DIR}/external/INMOST)

add_subdirectory(fem)
add_subdirectory(fvm)