#include "testfn.cuh"
#include <cuda_runtime.h>
#include <iostream>

__global__ void helloFromGPU() {
    printf("Hello from GPU!\n");
}

void call_hello_from_gpu() {
    helloFromGPU<<<1, 1>>>();
    cudaDeviceSynchronize();
}