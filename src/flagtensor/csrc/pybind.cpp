#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "torch/python.h"

#include "flagtensor/operators.h"

namespace py = pybind11;

PYBIND11_MODULE(c_operators, m) {
  m.doc() = "FlagTensor C++ operator bindings";

  // === Unary operators (28) ===
  m.def("abs", &flagtensor::abs, py::arg("input"),
        "Element-wise absolute value");
  m.def("acos", &flagtensor::acos, py::arg("input"),
        "Element-wise arccosine");
  m.def("acosh", &flagtensor::acosh, py::arg("input"),
        "Element-wise inverse hyperbolic cosine");
  m.def("asin", &flagtensor::asin, py::arg("input"),
        "Element-wise arcsine");
  m.def("asinh", &flagtensor::asinh, py::arg("input"),
        "Element-wise inverse hyperbolic sine");
  m.def("atan", &flagtensor::atan, py::arg("input"),
        "Element-wise arctangent");
  m.def("atanh", &flagtensor::atanh, py::arg("input"),
        "Element-wise inverse hyperbolic tangent");
  m.def("ceil", &flagtensor::ceil, py::arg("input"),
        "Element-wise ceiling");
  m.def("conj", &flagtensor::conj, py::arg("input"),
        "Element-wise conjugate");
  m.def("cos", &flagtensor::cos, py::arg("input"),
        "Element-wise cosine");
  m.def("cosh", &flagtensor::cosh, py::arg("input"),
        "Element-wise hyperbolic cosine");
  m.def("exp", &flagtensor::exp, py::arg("input"),
        "Element-wise exponential");
  m.def("floor", &flagtensor::floor, py::arg("input"),
        "Element-wise floor");
  m.def("identity", &flagtensor::identity, py::arg("input"),
        "Element-wise identity (passthrough)");
  m.def("log", &flagtensor::log, py::arg("input"),
        "Element-wise natural logarithm");
  m.def("mish", &flagtensor::mish, py::arg("input"),
        "Mish activation function");
  m.def("neg", &flagtensor::neg, py::arg("input"),
        "Element-wise negation");
  m.def("rcp", &flagtensor::rcp, py::arg("input"),
        "Element-wise reciprocal (1/x)");
  m.def("relu", &flagtensor::relu, py::arg("input"),
        "ReLU activation function");
  m.def("sigmoid", &flagtensor::sigmoid, py::arg("input"),
        "Sigmoid activation function");
  m.def("sin", &flagtensor::sin, py::arg("input"),
        "Element-wise sine");
  m.def("sinh", &flagtensor::sinh, py::arg("input"),
        "Element-wise hyperbolic sine");
  m.def("soft_plus", &flagtensor::soft_plus, py::arg("input"),
        "Softplus activation function");
  m.def("soft_sign", &flagtensor::soft_sign, py::arg("input"),
        "Softsign activation function");
  m.def("sqrt", &flagtensor::sqrt, py::arg("input"),
        "Element-wise square root");
  m.def("swish", &flagtensor::swish, py::arg("input"),
        "Swish (SiLU) activation function");
  m.def("tan", &flagtensor::tan, py::arg("input"),
        "Element-wise tangent");
  m.def("tanh", &flagtensor::tanh, py::arg("input"),
        "Element-wise hyperbolic tangent");

  // === Binary operators (4) ===
  m.def("add", &flagtensor::add, py::arg("a"), py::arg("b"),
        "Element-wise addition");
  m.def("mul", &flagtensor::mul, py::arg("a"), py::arg("b"),
        "Element-wise multiplication");
  m.def("max", &flagtensor::max, py::arg("a"), py::arg("b"),
        "Element-wise maximum");
  m.def("min", &flagtensor::min, py::arg("a"), py::arg("b"),
        "Element-wise minimum");

  // === Contraction operators (3) ===
  m.def("contraction", &flagtensor::contraction,
        py::arg("a"), py::arg("b"),
        py::arg("trans_a") = false, py::arg("trans_b") = false,
        "General tensor contraction (GEMM)");
  m.def("contraction_trinary", &flagtensor::contraction_trinary,
        py::arg("a"), py::arg("b"), py::arg("c"),
        "Three-input tensor contraction");
  m.def("elementwise_trinary", &flagtensor::elementwise_trinary,
        py::arg("a"), py::arg("b"), py::arg("c"),
        "Generic element-wise trinary: a * b + c");

  // === Sparse operators (1) ===
  m.def("block_sparse_contraction", &flagtensor::block_sparse_contraction,
        py::arg("a"), py::arg("b"),
        py::arg("a_block_desc"), py::arg("b_block_desc"),
        "Block-sparse tensor contraction (fallback dense mode)");
}