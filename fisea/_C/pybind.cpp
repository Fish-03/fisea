#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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

}