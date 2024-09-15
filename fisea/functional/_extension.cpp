#include <pybind11/pybind11.h>
#include "testfn.h"  // Always include this, regardless of CUDA availability

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "testfn.cuh"  // Only include CUDA-specific headers if CUDA is available
#endif

namespace py = pybind11;

PYBIND11_MODULE(_cpp_extension, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: fisea.functional._cpp_extension

        .. autosummary::
           :toctree: _generate
            add
            sub
            mul
            div
    )pbdoc";

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
}
