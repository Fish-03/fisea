// 這裡定義一些有的沒的函數
#pragma once

const int kCudaThreadsNum = 512;

inline int CudaGetBlocks(const int N)
{
    return (N + kCudaThreadsNum - 1) / kCudaThreadsNum;
}

#define CUDA_KERNEL_LOOP(i, n)                          \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n);                                       \
         i += blockDim.x * gridDim.x)

namespace fisea
{
    inline void __grapIdx(std::vector<int> &indices, int depth, int start, TensorBase *t)
    {
        if (depth == t->ndim() - 1)
        {
            for (int i = 0; i < t->get_shape()[depth]; i++)
            {
                indices[start + i] = start + i * t->get_stride()[depth];
            }
        }
        else
        {
            for (int i = 0; i < t->get_shape()[depth]; i++)
            {
                __grapIdx(indices, depth + 1, start + i * t->get_stride()[depth], t);
            }
        }
    }
}