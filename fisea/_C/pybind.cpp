#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "functional/bindings.h"
#include "tensor/bindings.h"
#include "type.h"

namespace py = pybind11;
// extern void init_functional(py::module_ &m);
// extern void init_tensor(py::module_ &m);
// extern int add(int i, int j);

PYBIND11_MODULE(_C, m) {
    m.doc() = "Project Root Module";  // 模块文档字符串

    
    // 绑定 Device 枚举类型
    py::enum_<fisea::Device>(m, "Device")
        .value("CPU", fisea::Device::CPU)
        .value("CUDA", fisea::Device::CUDA)
        .export_values();

    // 绑定 Dtype 枚举类型
    py::enum_<fisea::Dtype>(m, "Dtype")
        .value("FLOAT", fisea::Dtype::FLOAT)
        .value("INT", fisea::Dtype::INT)
        .export_values();

    // 创建 functional 子模块
    py::module_ m1 = m.def_submodule("functional", "Functional submodule");
    init_functional(m1);
    // m1.doc() = "Functional submodule";  // 子模块文档字符串
    // m1.def("add", &add, "A function which adds two numbers");
    
    py::module_ m2 = m.def_submodule("tensor", "Tensor submodule");
    init_tensor(m2);
    // m2.doc() = "Tensor submodule";  // 子模块文档字符串

    // // 创建 tensor 子模块

}