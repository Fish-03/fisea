// 這個主要為了生成.EXE檔案,用於DEBUG

#include <iostream>
#include <vector>
#include <typeinfo>
#include "FloatTensor.h"

int main() {
    std::vector<int> shape  {3, 4};

    // float data[24];
    // for (int i = 0; i < 24; i++) {
    //     data[i] = i;
    // }

    // auto dataPtr = std::shared_ptr<float>(data);
    // std::shared_ptr<float> dataPtr(new float[24], std::default_delete<float[]>());
    // for (int i = 0; i < 24; i++) {
    //     dataPtr.get()[i] = i;
    // }
    auto t = fisea::FloatTensor::create(shape);
    // std::cout << typeid(t).name() << std::endl;
    // t->set_data(dataPtr);
    std::cout << "==== " << std::endl;
    t->print();
    auto a = t->cpu();
    t->fill_(1);
    t->print();
    t->uniform_();
    t->print();
    t->normal_();
    t->print();
    std::cout << "==== " << std::endl;
    auto b = t->cuda();
    b->zeros_();
    b->print();
    b->ones_();
    b->print();
    b->uniform_();
    b->print();
    b->normal_();
    b->print();

    return 0;
}