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

# 定义一个递归查找 pybind.cpp 的函数并创建模块
function(make_c_submodules MODULE_PATH)
  # 遍历 fisea/_C 下所有子目录
  file(GLOB SUBMODULE_DIRS LIST_DIRECTORIES true "${MODULE_PATH}/*")
  
  foreach(SUBMODULE_DIR ${SUBMODULE_DIRS})
    # 检查每个子目录中是否有 pybind.cpp 文件
    if (EXISTS "${SUBMODULE_DIR}/pybind.cpp")
      # 替换目录名为模块名
      string(REPLACE "/" "_" MODULE_NAME ${SUBMODULE_DIR})

      # 找到该目录下的所有 .cpp 源文件
      file(GLOB_RECURSE SOURCES ${SUBMODULE_DIR}/*.cpp)

      # 如果找到了 CUDA 文件，添加它们到 SOURCES
      if (CUDAToolkit_FOUND)
        file(GLOB_RECURSE CUDA_SOURCES ${SUBMODULE_DIR}/*.cu)
        list(APPEND SOURCES ${CUDA_SOURCES})
      endif()

      # 如果有源文件，建立该模块
      if(SOURCES)
        add_library(${MODULE_NAME}_LIBC ${SOURCES})
        set_target_properties(${MODULE_NAME}_LIBC PROPERTIES POSITION_INDEPENDENT_CODE ON)

        # 添加 Python 模块并绑定 pybind.cpp
        python_add_library(${MODULE_NAME} MODULE "${SUBMODULE_DIR}/pybind.cpp" WITH_SOABI)

        # 链接 pybind11 头文件库
        target_link_libraries(${MODULE_NAME} PRIVATE pybind11::headers)

        # 如果找到 CUDA，链接 CUDA 运行时库并定义标志
        if(CUDAToolkit_FOUND)
          target_link_libraries(${MODULE_NAME} PRIVATE CUDA::cudart)
          target_compile_definitions(${MODULE_NAME} PRIVATE USE_CUDA)
        endif()

        # 链接 C++ 库
        target_link_libraries(${MODULE_NAME} PRIVATE ${MODULE_NAME}_LIBC)

        # 将模块添加到父模块
        list(APPEND ALL_SUBMODULES ${MODULE_NAME})
      endif()
    endif()
  endforeach()

  # 返回找到的所有子模块
  set(ALL_SUBMODULES ${ALL_SUBMODULES} PARENT_SCOPE)
endfunction()

# 主函数，用于创建顶层 _C 模块
function(make_c_module MODULE_PATH)
  # 调用上面的函数递归创建子模块
  make_c_submodules(${MODULE_PATH})

  # 创建顶层 _C 库
  if(ALL_SUBMODULES)
    python_add_library(_C MODULE "${CMAKE_CURRENT_SOURCE_DIR}/${MODULE_PATH}/pybind.cpp" WITH_SOABI)

    # 链接所有子模块到 _C
    foreach(SUBMODULE ${ALL_SUBMODULES})
      target_link_libraries(_C PRIVATE ${SUBMODULE})
    endforeach()

    message(STATUS "Created top-level module _C with submodules: ${ALL_SUBMODULES}")
  endif()
endfunction()


# ======================== Configurate _C ========================
# functional
make_c_module(fisea/_C)

# function(add_python_module MODULE_PATH)
#   string(REPLACE "/" "_" MODULE_NAME ${MODULE_PATH})
#   file(GLOB SOURCES ${MODULE_PATH}/*.cpp)

#   # Only add .cu files if CUDA is found
#   if (CUDAToolkit_FOUND)
#     file(GLOB CUDA_SOURCES ${MODULE_PATH}/*.cu)
#     list(APPEND SOURCES ${CUDA_SOURCES})
#   endif()

#   list(REMOVE_ITEM SOURCES "${PROJECT_SOURCE_DIR}/${MODULE_PATH}/_extension.cpp")

#   if(SOURCES)
#     add_library(${MODULE_NAME}_LIB ${SOURCES})
#     set_target_properties(${MODULE_NAME}_LIB PROPERTIES POSITION_INDEPENDENT_CODE ON)
#     python_add_library(${MODULE_NAME} MODULE "${PROJECT_SOURCE_DIR}/${MODULE_PATH}/_extension.cpp" WITH_SOABI)
#     set_target_properties(${MODULE_NAME} PROPERTIES 
#                           LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${MODULE_PATH} 
#                           OUTPUT_NAME "_cpp_extension"
#                           )
#     target_link_libraries(${MODULE_NAME} PRIVATE pybind11::headers)

#     # Link CUDA runtime and define CUDA flags if CUDA is found
#     if(CUDAToolkit_FOUND)
#       target_link_libraries(${MODULE_NAME} PRIVATE CUDA::cudart)
#       target_compile_definitions(${MODULE_NAME} PRIVATE USE_CUDA)  # Define flag for CUDA support
#     endif()

#     target_link_libraries(${MODULE_NAME} PRIVATE ${MODULE_NAME}_LIB)
#     install(TARGETS ${MODULE_NAME} DESTINATION ${MODULE_PATH})
#   endif()
# endfunction()

# # Add the modules
# add_python_module("fisea")
# add_python_module("fisea/functional")

