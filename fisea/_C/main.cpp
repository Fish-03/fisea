#include <iostream>
#include <vector>
#include <typeinfo>
#include "FloatTensor.h"
#include "FunctionBase.h"
#include "ReLU.h"
int main() {
    std::vector<int> shape  {3, 3};

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
    // t->print();
    auto a = t->cpu();
    t->fill_(1);
    t->print();
    t->uniform_();
    t->print();
    t->normal_();
    t->print();

    auto x = fisea::FloatTensor::create(shape);
    x->normal_();
    auto y = fisea::FloatTensor::create(shape);
    y->normal_();
    auto z = fisea::Add::apply(x, y);
    auto b = fisea::relu::apply(z);
    std::cout << "x: " << std::endl;
    x->print();
    std::cout << "y: " << std::endl;
    y->print();
    std::cout << "z: " << std::endl;
    z->print();
    std::cout << "result: " << std::endl;
    b->print();

    b->backward();
    
    std::cout << "x" << std::endl;
    x->print();
    std::cout << "x.grad: " << std::endl;
    if (x->get_grad() == nullptr) {
        std::cout << "grad is null" << std::endl;
    }
    else {
        x->get_grad()->print();
    }
    
    return 0;
}