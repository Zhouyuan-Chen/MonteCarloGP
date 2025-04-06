list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

cmake_minimum_required(VERSION 3.24)

project(TinyAlligator VERSION 0.1)

add_library(TinyAlligator)

include(eigen)
include(libigl)
igl_include(glfw)
# igl_include(imgui)
add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(tests)

add_executable(alligator_play main.cpp)

target_link_libraries(TinyAlligator PUBLIC
    Eigen3::Eigen
    igl::glfw
    )

target_link_libraries(TinyAlligator PUBLIC Eigen3::Eigen igl::glfw)
target_link_libraries(alligator_play PRIVATE TinyAlligator igl::glfw)

enable_testing()
