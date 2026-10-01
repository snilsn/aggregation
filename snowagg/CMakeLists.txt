cmake_minimum_required(VERSION 3.15)
project(snowagg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(SNOWAGG_NATIVE "Optimize for the build machine (-march=native)" ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenMP)

pybind11_add_module(_core src/bindings.cpp)
target_include_directories(_core PRIVATE include)
target_link_libraries(_core PRIVATE Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
  target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
endif()
# No fused multiply-add contraction and no fast-math: results must match the
# Python reference bit for bit where the algorithms allow it.
target_compile_options(_core PRIVATE -O3 -ffp-contract=off -fno-fast-math -Wall -Wextra)
if(SNOWAGG_NATIVE)
  target_compile_options(_core PRIVATE -march=native)
endif()

if(SKBUILD)
  install(TARGETS _core DESTINATION snowagg)
else()
  # development build: put the module next to the Python sources
  set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/snowagg)
endif()
