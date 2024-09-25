#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>   // 如果需要處理 numpy arrays
#include <pybind11/stl.h>     // 如果需要處理 STL 容器 (例如 std::vector)

#include "../type.h"
#include "Tensor.h"           // 引入 Tensor 類的定義

// namespace py = pybind11;

inline void init_tensor(py::module_ &m) {
    py::class_<fisea::Tensor>(m, "Tensor")
        .def(py::init<fisea::Shape, fisea::Device, fisea::Dtype, void *>(),
             py::arg("shape"), py::arg("device") = fisea::Device::CPU, py::arg("dtype") = fisea::Dtype::FLOAT, py::arg("data") = nullptr)
        .def(py::init<fisea::Shape, std::string, fisea::Dtype, void *>(),
             py::arg("shape"), py::arg("device") = "cpu", py::arg("dtype") = fisea::Dtype::FLOAT, py::arg("data") = nullptr)
        .def(py::init<fisea::Shape, fisea::Device, std::string, void *>(),
             py::arg("shape"), py::arg("device") = fisea::Device::CPU, py::arg("dtype") = "float", py::arg("data") = nullptr)
        .def(py::init<fisea::Shape, std::string, std::string, void *>(),
             py::arg("shape"), py::arg("device") = "cpu", py::arg("dtype") = "float", py::arg("data") = nullptr)
             
        .def_static("from", &fisea::Tensor::from)
        .def("copy", &fisea::Tensor::copy)
        .def("cpu", &fisea::Tensor::cpu)
        .def("cuda", &fisea::Tensor::cuda)

        .def("fill_", py::overload_cast<int>(&fisea::Tensor::fill_))
        .def("fill_", py::overload_cast<float>(&fisea::Tensor::fill_))
        
        .def("zero_", &fisea::Tensor::zero_)
        .def("one_", &fisea::Tensor::one_)
        .def("randn_", &fisea::Tensor::randn_)

        .def("int_", &fisea::Tensor::to_int_)
        .def("float_", &fisea::Tensor::to_float_)

        .def("int", &fisea::Tensor::to_int)
        .def("float", &fisea::Tensor::to_float)

        .def_static("zeros", py::overload_cast<fisea::Shape, fisea::Device, fisea::Dtype>(&fisea::Tensor::zeros),
                    py::arg("shape"), py::arg("device") = fisea::Device::CPU, py::arg("dtype") = fisea::Dtype::FLOAT)
        .def_static("zeros", py::overload_cast<fisea::Shape, std::string, fisea::Dtype>(&fisea::Tensor::zeros),
                    py::arg("shape"), py::arg("device") = "cpu", py::arg("dtype") = fisea::Dtype::FLOAT)
        .def_static("zeros", py::overload_cast<fisea::Shape, fisea::Device, std::string>(&fisea::Tensor::zeros),
                    py::arg("shape"), py::arg("device") = fisea::Device::CPU, py::arg("dtype") = "float")
        .def_static("zeros", py::overload_cast<fisea::Shape, std::string, std::string>(&fisea::Tensor::zeros),
                    py::arg("shape"), py::arg("device") = "cpu", py::arg("dtype") = "float")

        .def_static("ones", py::overload_cast<fisea::Shape, fisea::Device, fisea::Dtype>(&fisea::Tensor::ones),
                    py::arg("shape"), py::arg("device") = fisea::Device::CPU, py::arg("dtype") = fisea::Dtype::FLOAT)
        .def_static("ones", py::overload_cast<fisea::Shape, std::string, fisea::Dtype>(&fisea::Tensor::ones),
                    py::arg("shape"), py::arg("device") = "cpu", py::arg("dtype") = fisea::Dtype::FLOAT)
        .def_static("ones", py::overload_cast<fisea::Shape, fisea::Device, std::string>(&fisea::Tensor::ones),
                    py::arg("shape"), py::arg("device") = fisea::Device::CPU, py::arg("dtype") = "float")
        .def_static("ones", py::overload_cast<fisea::Shape, std::string, std::string>(&fisea::Tensor::ones),
                    py::arg("shape"), py::arg("device") = "cpu", py::arg("dtype") = "float")

        .def_static("randn", py::overload_cast<fisea::Shape, fisea::Device, fisea::Dtype>(&fisea::Tensor::randn),
                    py::arg("shape"), py::arg("device") = fisea::Device::CPU, py::arg("dtype") = fisea::Dtype::FLOAT)
        .def_static("randn", py::overload_cast<fisea::Shape, std::string, fisea::Dtype>(&fisea::Tensor::randn),
                    py::arg("shape"), py::arg("device") = "cpu", py::arg("dtype") = fisea::Dtype::FLOAT)
        .def_static("randn", py::overload_cast<fisea::Shape, fisea::Device, std::string>(&fisea::Tensor::randn),
                    py::arg("shape"), py::arg("device") = fisea::Device::CPU, py::arg("dtype") = "float")
        .def_static("randn", py::overload_cast<fisea::Shape, std::string, std::string>(&fisea::Tensor::randn),
                    py::arg("shape"), py::arg("device") = "cpu", py::arg("dtype") = "float");
}
