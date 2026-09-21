add_library(operators
            SHARED
            utils.cpp
            unary.cpp
            binary.cpp
            contraction.cpp
            sparse.cpp)

target_include_directories(operators
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(operators
  PUBLIC Torch::Torch
  PRIVATE TritonJIT::triton_jit
  PRIVATE CUDA::cudart CUDA::cuda_driver)

target_compile_definitions(operators PRIVATE BACKEND_${FLAGTENSOR_BACKEND})

add_library(FlagTensor::operators ALIAS operators)

set_target_properties(operators PROPERTIES
  EXPORT_NAME operators
  OUTPUT_NAME flagtensor
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR})

# Install the triton_src directory alongside the shared library
if(FLAGTENSOR_INSTALL)
  include(GNUInstallDirs)
  install(
    DIRECTORY triton_src/
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/flagtensor/triton_src)
  install(TARGETS operators
    EXPORT FlagTensorTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()
