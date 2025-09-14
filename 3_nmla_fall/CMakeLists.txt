cmake_minimum_required(VERSION 3.22)

project(numerical_methods_of_linear_algebra CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-pedantic")

set(LIBRARY SRC_LIB)
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(benchmark)