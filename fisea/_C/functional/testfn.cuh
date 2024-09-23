#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void helloFromGPU();
void call_hello_from_gpu();
