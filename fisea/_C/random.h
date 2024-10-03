#pragma once

#include <random>

namespace fisea {
    std::random_device randDevice;
    std::mt19937 randGen(randDevice());

    float rand(float min, float max) {
        std::uniform_real_distribution<float> dis(min, max);
        return dis(randGen);
    }

    float randn() {
        std::normal_distribution<float> dis(0, 1);
        return dis(randGen);
    }
}