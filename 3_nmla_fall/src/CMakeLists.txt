cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-pedantic")

add_library(${LIBRARY} SHARED)

target_sources(${LIBRARY}
    PUBLIC
        matrix.cpp
        vector.cpp
        triangular_matrix.cpp
        givens_rotations.cpp
        chebyshev_iter.cpp
)