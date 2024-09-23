#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>   // 如果需要處理 numpy arrays
#include <pybind11/stl.h>     // 如果需要處理 STL 容器 (例如 std::vector)

#include "../type.h"
#include "Tensor.h"           // 引入 Tensor 類的定義

namespace py = pybind11;

inline void init_tensor(py::module &m) {
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

    // 绑定 Shape 类（假设 Shape 是一个 std::vector<int>）
    py::class_<fisea::Shape>(m, "Shape")
        .def(py::init<>())
        .def(py::init<const std::vector<int>&>())
        .def("size", &fisea::Shape::size)
        .def("__getitem__", [](const fisea::Shape &s, int i) {
            return s[i];
        })
        .def("__iter__", [](const fisea::Shape &s) {
            return py::make_iterator(s.begin(), s.end());
        }, py::keep_alive<0, 1>());

    // 绑定 Tensor 类
    py::class_<fisea::Tensor, std::shared_ptr<fisea::Tensor>>(m, "Tensor")
        // 构造函数重载
        .def(py::init<fisea::Shape, void*, fisea::Device, fisea::Dtype>(),
             py::arg("shape"), py::arg("data") = nullptr,
             py::arg("device") = fisea::Device::CPU,
             py::arg("dtype") = fisea::Dtype::FLOAT)
        .def(py::init<fisea::Shape, void*, std::string, fisea::Dtype>(),
             py::arg("shape"), py::arg("data") = nullptr,
             py::arg("device") = "cpu",
             py::arg("dtype") = fisea::Dtype::FLOAT)
        .def(py::init<fisea::Shape, void*, fisea::Device, std::string>(),
             py::arg("shape"), py::arg("data") = nullptr,
             py::arg("device") = fisea::Device::CPU,
             py::arg("dtype") = "float")
        .def(py::init<fisea::Shape, void*, std::string, std::string>(),
             py::arg("shape"), py::arg("data") = nullptr,
             py::arg("device") = "cpu",
             py::arg("dtype") = "float")

        // 静态方法
        .def_static("from", &fisea::Tensor::from)

        // 成员方法
        .def("copy", &fisea::Tensor::copy)
        .def("cpu", &fisea::Tensor::cpu)
        .def("cuda", &fisea::Tensor::cuda)
        .def("to_int", &fisea::Tensor::to_int)
        .def("to_float", &fisea::Tensor::to_float)

        // 重载 fill_ 方法
        .def("fill_", py::overload_cast<int>(&fisea::Tensor::fill_))
        .def("fill_", py::overload_cast<float>(&fisea::Tensor::fill_))

        .def("zero_", &fisea::Tensor::zero_)
        .def("one_", &fisea::Tensor::one_)
        .def("randn_", &fisea::Tensor::randn_)

        // 静态方法 zeros
        .def_static("zeros",
            py::overload_cast<fisea::Shape, fisea::Device, fisea::Dtype>(&fisea::Tensor::zeros),
            py::arg("shape"),
            py::arg("device") = fisea::Device::CPU,
            py::arg("dtype") = fisea::Dtype::FLOAT)
        .def_static("zeros",
            py::overload_cast<fisea::Shape, std::string, fisea::Dtype>(&fisea::Tensor::zeros),
            py::arg("shape"),
            py::arg("device") = "cpu",
            py::arg("dtype") = fisea::Dtype::FLOAT)
        .def_static("zeros",
            py::overload_cast<fisea::Shape, fisea::Device, std::string>(&fisea::Tensor::zeros),
            py::arg("shape"),
            py::arg("device") = fisea::Device::CPU,
            py::arg("dtype") = "float")
        .def_static("zeros",
            py::overload_cast<fisea::Shape, std::string, std::string>(&fisea::Tensor::zeros),
            py::arg("shape"),
            py::arg("device") = "cpu",
            py::arg("dtype") = "float")

        // 静态方法 ones
        .def_static("ones",
            py::overload_cast<fisea::Shape, fisea::Device, fisea::Dtype>(&fisea::Tensor::ones),
            py::arg("shape"),
            py::arg("device") = fisea::Device::CPU,
            py::arg("dtype") = fisea::Dtype::FLOAT)
        .def_static("ones",
            py::overload_cast<fisea::Shape, std::string, fisea::Dtype>(&fisea::Tensor::ones),
            py::arg("shape"),
            py::arg("device") = "cpu",
            py::arg("dtype") = fisea::Dtype::FLOAT)
        .def_static("ones",
            py::overload_cast<fisea::Shape, fisea::Device, std::string>(&fisea::Tensor::ones),
            py::arg("shape"),
            py::arg("device") = fisea::Device::CPU,
            py::arg("dtype") = "float")
        .def_static("ones",
            py::overload_cast<fisea::Shape, std::string, std::string>(&fisea::Tensor::ones),
            py::arg("shape"),
            py::arg("device") = "cpu",
            py::arg("dtype") = "float")

        // 静态方法 randn
        .def_static("randn",
            py::overload_cast<fisea::Shape, fisea::Device, fisea::Dtype>(&fisea::Tensor::randn),
            py::arg("shape"),
            py::arg("device") = fisea::Device::CPU,
            py::arg("dtype") = fisea::Dtype::FLOAT)
        .def_static("randn",
            py::overload_cast<fisea::Shape, std::string, fisea::Dtype>(&fisea::Tensor::randn),
            py::arg("shape"),
            py::arg("device") = "cpu",
            py::arg("dtype") = fisea::Dtype::FLOAT)
        .def_static("randn",
            py::overload_cast<fisea::Shape, fisea::Device, std::string>(&fisea::Tensor::randn),
            py::arg("shape"),
            py::arg("device") = fisea::Device::CPU,
            py::arg("dtype") = "float")
        .def_static("randn",
            py::overload_cast<fisea::Shape, std::string, std::string>(&fisea::Tensor::randn),
            py::arg("shape"),
            py::arg("device") = "cpu",
            py::arg("dtype") = "float")

        // 可根据需要添加更多方法
        ;
}