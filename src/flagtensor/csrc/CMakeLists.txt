pybind11_add_module(c_operators pybind.cpp)
target_link_libraries(c_operators PUBLIC Torch::Torch_Python PRIVATE operators)
set_target_properties(c_operators PROPERTIES
  INSTALL_RPATH "${_rpath_portable_origin}/${CMAKE_INSTALL_LIBDIR}"
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON)

if(FLAGTENSOR_INSTALL)
  install(TARGETS c_operators
          LIBRARY DESTINATION ".")
endif()