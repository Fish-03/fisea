#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>   // 如果需要處理 numpy arrays
#include <pybind11/stl.h>     // 如果需要處理 STL 容器 (例如 std::vector)
#include "../type.h"
#include "tensor.h"           // 引入 Tensor 類的定義

#if USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

namespace py = pybind11;

void init_tensor(py::module_ &m) {
    py::class_<fisea::Tensor>(m, "Tensor")
        .def(py::init<fisea::Shape, void*, fisea::Device, fisea::Dtype>(),
            py::arg("shape"), py::arg("data") = nullptr, py::arg("device") = fisea::Device::CPU, py::arg("dtype") = fisea::Dtype::FLOAT)

        .def("copy", &fisea::Tensor::copy)
        .def("cpu", &fisea::Tensor::cpu)
        .def("cuda", &fisea::Tensor::cuda)
        .def("to_int", &fisea::Tensor::to_int)
        .def("to_float", &fisea::Tensor::to_float)

        .def_static("from", &fisea::Tensor::from)
        .def_static("zeros", &fisea::Tensor::zeros<fisea::Device, fisea::Dtype>,
                    py::arg("shape"), py::arg("device") = fisea::Device::CPU, py::arg("dtype") = fisea::Dtype::FLOAT)

        .def_static("ones", &fisea::Tensor::ones<fisea::Device, fisea::Dtype>,
                    py::arg("shape"), py::arg("device") = fisea::Device::CPU, py::arg("dtype") = fisea::Dtype::FLOAT)

        .def_static("randn", &fisea::Tensor::randn<fisea::Device, fisea::Dtype>,
                    py::arg("shape"), py::arg("device") = fisea::Device::CPU, py::arg("dtype") = fisea::Dtype::FLOAT)

        .def("fill_", [](fisea::Tensor &tensor, py::object value) {
            if (py::isinstance<py::float_>(value)) {
                tensor.fill_(value.cast<float>());
            } else if (py::isinstance<py::int_>(value)) {
                tensor.fill_(value.cast<int>());
            } else {
                throw std::runtime_error("Unsupported data type for fill_");
            }
        });
}