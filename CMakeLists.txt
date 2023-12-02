cmake_minimum_required(VERSION 3.12)
project(simbackend)

find_package(pybind11 REQUIRED)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)

add_library(simbackend MODULE simbackend.cpp)
include_directories("/usr/include/eigen3/")
target_link_libraries(simbackend PRIVATE pybind11::pybind11)

target_compile_options(simbackend PRIVATE -O2)