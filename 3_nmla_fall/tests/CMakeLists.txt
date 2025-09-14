cmake_minimum_required(VERSION 3.22)

project(tests CXX)
set(CMAKE_CXX_FLAGS "-pedantic")

enable_testing()

include_directories("${PROJECT_SOURCE_DIR}/../src")

find_package(GTest CONFIG REQUIRED)
include_directories(${GTest_INCLUDE_DIRS})

# include(FetchContent)
# FetchContent_Declare(
#     googletest
#     URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
#     )
# # # For Windows: Prevent overriding the parent project's compiler/linker settings
# # set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# FetchContent_MakeAvailable(googletest)
    
add_executable(
    ${PROJECT_NAME}
    test.cpp
)

# target_link_libraries(${PROJECT_NAME} GTest::gtest_main)
target_link_libraries(${PROJECT_NAME} ${LIBRARY})
target_link_libraries(${PROJECT_NAME} gtest::gtest)
# include(GoogleTest)