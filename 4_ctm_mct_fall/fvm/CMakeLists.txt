find_package(INMOST REQUIRED)
add_executable(fvm diffusion_fvm.cpp)
target_link_libraries(fvm INMOST::inmost)
