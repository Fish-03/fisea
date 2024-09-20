#include <pybind11/pybind11.h>
#include "testfn.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "testfn.cuh"
#endif

namespace py = pybind11;
void init_module(py::module_ &m) {
    m.doc() = "";

    // Register the basic math functions
    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("sub", &sub, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

    m.def("mul", &mul, R"pbdoc(
        Multiply two numbers

        Some other explanation about the multiply function.
    )pbdoc");

    m.def("div", &m_div, R"pbdoc(
        Divide two numbers

        Some other explanation about the divide function.
    )pbdoc");

#ifdef USE_CUDA
    // Register the CUDA function only if CUDA is available
    m.def("cuda_test", &call_hello_from_gpu, R"pbdoc(
        Hello from CUDA!
    )pbdoc");
#endif
    // 绑定更多 functional.cpp 中的函数或类
}
