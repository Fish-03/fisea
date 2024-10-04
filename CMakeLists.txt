# Require CMake 3.15+ (matching scikit-build-core)
# Use new versions of all policies up to CMake 3.27
cmake_minimum_required(VERSION 3.15...3.27)

# Scikit-build-core sets these values for you, or you can just hard-code the
# name and version.
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION})

set(PROJECT_LANGUAGES CXX)  # Start with just CXX as the default language

find_package(CUDAToolkit)

if(CUDAToolkit_FOUND)
  list(APPEND PROJECT_LANGUAGES CUDA)
  enable_language(CUDA)  # Explicitly enable CUDA language
endif()

# Find the module development requirements (requires FindPython from 3.17 or
# scikit-build-core's built-in backport)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

if(CUDAToolkit_FOUND)
  file(GLOB_RECURSE SOURCES "fisea/_C/*.cpp")
  list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/fisea/_C/main.cpp")
  file(GLOB_RECURSE CUDA_SOURCES "fisea/_C/*.cu")
  file(GLOB_RECURSE H SOURCES "fisea/_C/*.h")

    
  python_add_library(_C MODULE ${SOURCES} ${CUDA_SOURCES} WITH_SOABI)

  target_compile_definitions(_C PRIVATE USE_CUDA)
  set_target_properties(_C PROPERTIES POSITION_INDEPENDENT_CODE ON OUTPUT_NAME "_C")
  target_link_libraries(_C PRIVATE pybind11::headers)
  target_link_libraries(_C PRIVATE CUDA::cudart)

  file(GLOB_RECURSE SOURCES "fisea/_C/*.cpp")
  list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/fisea/_C/pybind.cpp")

  add_executable(_C_exe "fisea/_C/main.cpp" ${SOURCES} ${CUDA_SOURCES})

  target_compile_definitions(_C_exe PRIVATE USE_CUDA)
  target_link_libraries(_C_exe PRIVATE CUDA::cudart)

  # if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  #   target_compile_definitions(_C_exe PRIVATE DEBUG)
  # endif()
  
else()
  file(GLOB_RECURSE SOURCES "fisea/_C/*.cpp")
  file(GLOB_RECURSE H SOURCES "fisea/_C/*.h")
  python_add_library(_C MODULE ${SOURCES} WITH_SOABI)
  set_target_properties(_C PROPERTIES POSITION_INDEPENDENT_CODE ON OUTPUT_NAME "_C")
  target_link_libraries(_C PRIVATE pybind11::headers)

  add_executable(_C_exe ${SOURCES})
  target_link_libraries(_C_exe PRIVATE pybind11::headers)

  if(CMAKE_BUILD_TYPE STREQUAL "_DEBUG")
    target_compile_definitions(_C_exe PRIVATE DEBUG)
  endif()

endif()

target_compile_definitions(_C PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS _C DESTINATION fisea)

# ====== ADDITION FOR EXECUTABLE GENERATION ======

# Create an executable from main.cpp
# file(GLOB_RECURSE MAIN_SOURCES "fisea/_C/main.cpp")

# if(CUDAToolkit_FOUND)
#   add_executable(fisea_exe ${MAIN_SOURCES} ${CUDA_SOURCES})
#   target_link_libraries(fisea_exe PRIVATE CUDA::cudart _CU)
# else()
#   add_executable(fisea_exe ${MAIN_SOURCES})
# endif()

# # Link pybind11 and other necessary libraries if required
# target_link_libraries(fisea_exe PRIVATE pybind11::headers)

# # Define properties for the executable
# set_target_properties(fisea_exe PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
#     OUTPUT_NAME "fisea_exe"
# )