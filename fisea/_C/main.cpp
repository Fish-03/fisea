// 這個主要為了生成.EXE檔案,用於DEBUG

#include <iostream>
#include <vector>
#include "functional/kernel.cuh"
#include "tensor/Tensor.h"

int main() {
    std::vector<int> shape = {2, 3, 4};
    fisea::Tensor t = fisea::Tensor(shape, "cpu", "float");
    t.cpu();
    // t.cuda();
    return 0;
}