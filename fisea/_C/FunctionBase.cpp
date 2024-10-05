#include "FunctionBase.h"

using namespace fisea;

FunctionBase::~FunctionBase(){}

std::shared_ptr<FloatTensor> FunctionBase::operator()(std::shared_ptr<FloatTensor> input, std::shared_ptr<FloatTensor> output)
{
    return forward(input, output);
}