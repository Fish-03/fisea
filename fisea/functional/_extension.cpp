#include <pybind11/pybind11.h>
#include "testfn.h"

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
            hello_from_gpu

    )pbdoc";

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

}
