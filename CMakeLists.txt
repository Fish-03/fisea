# Require CMake 3.15+ (matching scikit-build-core)
# Use new versions of all policies up to CMake 3.27
cmake_minimum_required(VERSION 3.15...3.27)

# Scikit-build-core sets these values for you, or you can just hard-code the
# name and version.
project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX CUDA)

# Find the module development requirements (requires FindPython from 3.17 or
# scikit-build-core's built-in backport)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Try to find CUDA. If found, it will be used; otherwise, continue without CUDA.
find_package(CUDAToolkit)

function(add_python_module MODULE_PATH)
  string(REPLACE "/" "_" MODULE_NAME ${MODULE_PATH})
  file(GLOB SOURCES ${MODULE_PATH}/*.cpp)

  # Only add .cu files if CUDA is found
  if (CUDAToolkit_FOUND)
    file(GLOB CUDA_SOURCES ${MODULE_PATH}/*.cu)
    list(APPEND SOURCES ${CUDA_SOURCES})
  endif()

  list(REMOVE_ITEM SOURCES "${PROJECT_SOURCE_DIR}/${MODULE_PATH}/_extension.cpp")

  if(SOURCES)
    add_library(${MODULE_NAME}_LIB ${SOURCES})
    set_target_properties(${MODULE_NAME}_LIB PROPERTIES POSITION_INDEPENDENT_CODE ON)
    python_add_library(${MODULE_NAME} MODULE "${PROJECT_SOURCE_DIR}/${MODULE_PATH}/_extension.cpp" WITH_SOABI)
    set_target_properties(${MODULE_NAME} PROPERTIES 
                          LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${MODULE_PATH} 
                          OUTPUT_NAME "_cpp_extension"
                          )
    target_link_libraries(${MODULE_NAME} PRIVATE pybind11::headers)

    # Link CUDA runtime and define CUDA flags if CUDA is found
    if(CUDAToolkit_FOUND)
      target_link_libraries(${MODULE_NAME} PRIVATE CUDA::cudart)
      target_compile_definitions(${MODULE_NAME} PRIVATE USE_CUDA)  # Define flag for CUDA support
    endif()

    target_link_libraries(${MODULE_NAME} PRIVATE ${MODULE_NAME}_LIB)
    install(TARGETS ${MODULE_NAME} DESTINATION ${MODULE_PATH})
  endif()
endfunction()

# Add the modules
add_python_module("fisea")
add_python_module("fisea/functional")
