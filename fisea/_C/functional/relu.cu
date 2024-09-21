// #include <cuda_runtime.h> // Why this is not neccessary?
// #include <device_launch_parameters.h>
#include <iostream>

#include "../handler.cuh"

__global__ void relu_gpu(float* in, float* out, int n) {
  CUDA_KERNEL_LOOP(i, n) {
  out[i] = in[i] > 0 ? in[i] : 0;
  }
}
