#pragma once

#ifndef USE_CUDA
#define USE_CUDA false
#endif

namespace fisea {
    std::random_device rd;
    int random_seed = rd();
}