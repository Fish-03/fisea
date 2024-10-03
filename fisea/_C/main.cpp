// 這個主要為了生成.EXE檔案,用於DEBUG

#include <iostream>
#include <vector>
#include <typeinfo>
#include "FloatTensor.h"

int main() {
    std::vector<int> shape  {int(2), int(3), int(4)};

    // float data[24];
    // for (int i = 0; i < 24; i++) {
    //     data[i] = i;
    // }

    // auto dataPtr = std::shared_ptr<float>(data);
    std::shared_ptr<float> dataPtr(new float[24], std::default_delete<float[]>());
    for (int i = 0; i < 24; i++) {
        dataPtr.get()[i] = i;
    }
    std::shared_ptr<fisea::FloatTensor> t = std::make_shared<fisea::FloatTensor>(shape);
    std::cout << typeid(t).name() << std::endl;
    t->set_data(dataPtr);
    std::cout << "==== " << std::endl;
    t->print();
    auto a = t->cpu();
    
    // t.cuda();
    return 0;
}