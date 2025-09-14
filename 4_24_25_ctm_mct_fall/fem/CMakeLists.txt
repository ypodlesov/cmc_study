find_package(INMOST REQUIRED)
add_executable(fem diffusion_fem.cpp)
target_link_libraries(fem INMOST::inmost)
