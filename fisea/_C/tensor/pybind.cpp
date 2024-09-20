#include <pybind11/pybind11.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#endif

void init_module(py::module_ &m) {
    m.doc() = "";
}