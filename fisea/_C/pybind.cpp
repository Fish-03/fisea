#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

namespace py = pybind11;

void init_module(py::module_ &);

PYBIND11_MODULE(project_root, m) {
    m.doc() = "Project Root Module";  // 模块文档字符串

    // 创建 functional 子模块
    py::module_ functional_module = m.def_submodule("functional", "Functional submodule");
    init_module(functional_module);

    // 创建 tensor 子模块
    py::module_ tensor_module = m.def_submodule("tensor", "Tensor submodule");
    init_module(tensor_module);
}