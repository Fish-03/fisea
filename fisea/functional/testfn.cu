#include <cuda_runtime.h>
#include <iostream>

__global__ void helloFromGPU() {
    printf("Hello from GPU!\n");
}

// 调用 CUDA kernel 的 C++ 包装函数
void call_hello_from_gpu() {
    // 在 GPU 上启动 kernel
    helloFromGPU<<<1, 1>>>();

    // 等待 GPU 完成任务
    cudaDeviceSynchronize();
}