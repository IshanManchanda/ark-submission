cmake_minimum_required(VERSION 3.17)
project(task1)

set(CMAKE_CXX_STANDARD 14)

add_executable(task1 main.cpp)

target_compile_options(task1 PRIVATE -fopenmp -O3)
