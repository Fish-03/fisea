#pragma once

#include <pybind11/pybind11.h>

#include "testfn.h"
#include "../const.h"
#include "testfn.cuh"

namespace py = pybind11;

inline void init_functional(py::module_ &m)
{
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

    // Register the CUDA function only if CUDA is available
    m.def("cuda_test", &call_hello_from_gpu, R"pbdoc(
        Hello from CUDA!
    )pbdoc");
    // 绑定更多 functional.cpp 中的函数或类
}
