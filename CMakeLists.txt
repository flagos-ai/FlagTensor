cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(FlagTensor LANGUAGES CXX CUDA VERSION 0.1.0)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

# ------------------------------- project-wide settings -------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --------------------------- RPATH settings ---------------------------
if(APPLE)
  set(CMAKE_MACOSX_RPATH ON)
  set(_rpath_portable_origin "@loader_path")
else()
  set(_rpath_portable_origin $ORIGIN)
endif()

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${_rpath_portable_origin}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# --------------------------- project options ---------------------------
option(FLAGTENSOR_BUILD_C_EXTENSIONS "whether to build c extensions" ON)
option(FLAGTENSOR_USE_EXTERNAL_TRITON_JIT
       "whether to use an installed TritonJIT package" OFF)
option(FLAGTENSOR_INSTALL "Whether to install the package" ${PROJECT_IS_TOP_LEVEL})
set(FLAGTENSOR_BACKEND "CUDA" CACHE STRING "Target backend: CUDA")
set_property(CACHE FLAGTENSOR_BACKEND PROPERTY STRINGS "CUDA")
set(FLAGTENSOR_PYTHON_INSTALL_DIR "." CACHE STRING
    "Install directory for the c_operators Python extension")

if(NOT FLAGTENSOR_BUILD_C_EXTENSIONS)
  return()
endif()

message(STATUS "Building FlagTensor C++ wrapper with backend: ${FLAGTENSOR_BACKEND}")

# ------------------------------- dependencies -------------------------------
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(CUDAToolkit REQUIRED)
find_package(Torch MODULE REQUIRED)

# pybind11
execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_ROOT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED)

# TritonJIT
if(FLAGTENSOR_USE_EXTERNAL_TRITON_JIT)
  find_package(TritonJIT 0.1.0 CONFIG REQUIRED)
  if(DEFINED TritonJIT_BACKEND AND
     NOT TritonJIT_BACKEND STREQUAL FLAGTENSOR_BACKEND)
    message(FATAL_ERROR
      "FlagTensor backend ${FLAGTENSOR_BACKEND} requires a matching "
      "TritonJIT package, but found ${TritonJIT_BACKEND}")
  endif()
else()
  include(FetchContent)
  set(BACKEND ${FLAGTENSOR_BACKEND} CACHE STRING "Backend for TritonJIT" FORCE)
  FetchContent_Declare(TritonJIT
    URL https://resource.flagos.net/repository/flagos-filestore/libtriton_jit/libtriton_jit-0.2.0-rc0.1.tar.gz
  )
  FetchContent_MakeAvailable(TritonJIT)
endif()

# --------------------------- subdirectories ---------------------------
add_subdirectory(lib)
add_subdirectory(src/flagtensor/csrc)

# --------------------------- install package ---------------------------
if(FLAGTENSOR_INSTALL)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)
  install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/flagtensor"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

  configure_package_config_file(
    cmake/FlagTensorConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/FlagTensorConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FlagTensor)
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/FlagTensorConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
  install(EXPORT FlagTensorTargets
    FILE FlagTensorTargets.cmake
    NAMESPACE FlagTensor::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FlagTensor)
  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/FlagTensorConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/FlagTensorConfigVersion.cmake"
    cmake/FindTorch.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FlagTensor)
endif()
