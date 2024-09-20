#include <pybind11/pybind11.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#endif

namespace py = pybind11;

void init_tensor(py::module_ &m) {
    m.doc() = "";
}
