// #include <cuda_runtime.h> // Why this is not neccessary?
// #include <device_launch_parameters.h>
#include <iostream>

#include "testfn.cuh"

__global__ void helloFromGPU() {
    printf("Hello from GPU!\n");
}

void call_hello_from_gpu() {
    helloFromGPU<<<1, 1>>>();
    cudaDeviceSynchronize();
}