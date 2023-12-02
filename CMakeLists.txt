cmake_minimum_required(VERSION 3.12)
project(simbackend)

# Specify the path to the Python executable
set(Python3_EXECUTABLE "/home/daniel/anaconda3/envs/py311/bin/python") # This will give you errors, replace with your own path!!!

# Specify the Python version
find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter Development)

find_package(pybind11 REQUIRED)

# Include Python headers
include_directories(${Python3_INCLUDE_DIRS})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)

add_library(simbackend MODULE simbackend.cpp)
include_directories("/usr/include/eigen3/")
target_link_libraries(simbackend PRIVATE pybind11::pybind11)

target_compile_options(simbackend PRIVATE -O2)