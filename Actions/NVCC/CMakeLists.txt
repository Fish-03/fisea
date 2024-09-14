cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(simplest LANGUAGES CXX CUDA)

add_executable(simplest ../simplest.cu)
set_target_properties(simplest PROPERTIES LINKER_LANGUAGE CXX)

## CMAKE_CUDA_ARCHITECTURES now detected for NVCC, empty CUDA_ARCHITECTURES not allowed.
set_property(TARGET simplest PROPERTY CUDA_ARCHITECTURES OFF) 

enable_testing()
add_test(NAME simplest COMMAND simplest) 
