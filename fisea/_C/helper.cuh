#pragma once
#include <curand.h>

namespace fisea {
    curandGenerator_t gen;
    bool isInitialized = false;

    inline void initializeGenerator(unsigned long long seed = 1234ULL) {
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

        // 設置種子
        curandSetPseudoRandomGeneratorSeed(gen, seed);
        isInitialized = true;
    }



    inline void cudaUniform(float* data, size_t N) {
        if (!isInitialized) {
            initializeGenerator();
        }
        // 生成均勻分布的隨機數 [0, 1) 在 GPU 上
        auto error = curandGenerateUniform(gen, data, N);
        if (error != CURAND_STATUS_SUCCESS) {
            std::cerr << "Error: " << error << std::endl;
        }
    }

    inline void cudaNormal(float* data, size_t N, float mean = 0.0, float std = 1.0) {
        if (!isInitialized) {
            initializeGenerator();
        }
        // 生成正態分布的隨機數 N(mean, stddev^2) 在 GPU 上
        auto error = curandGenerateNormal(gen, data, N, mean, std);
        if (error != CURAND_STATUS_SUCCESS) {
            std::cerr << "Error: " << error << std::endl;
        }
    }
}
