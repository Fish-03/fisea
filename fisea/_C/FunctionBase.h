
#pragma once
#include <iostream>
#include <variant>
#include <tuple>
#include <map>
#include <memory>
#include <functional>
#include <typeinfo>

#include <utility>
#include <cstddef>

#include "type.h"
#include "FloatTensor.h"
namespace fisea
{
    class FunctionBase
    {
    private:
        virtual std::shared_ptr<FloatTensor> forward(std::shared_ptr<FloatTensor> input, std::shared_ptr<FloatTensor> output = nullptr) = 0;
        virtual void backward(std::shared_ptr<FloatTensor> input, std::shared_ptr<FloatTensor> grad = nullptr) = 0;

    public:
        virtual ~FunctionBase();
        // virtual std::shared_ptr<Tensor> backward(std::shared_ptr<Tensor> grad) = 0;
        std::shared_ptr<FloatTensor> operator()(std::shared_ptr<FloatTensor> input, std::shared_ptr<FloatTensor> output = nullptr);
    };

    template <typename T>
    class FnBase
    {

    public:
        template <typename U = T, typename... Args>
        static auto apply(Args... args) -> decltype(U::forward(std::declval<ctx_t &>(), std::forward<Args>(args)...))
        {
            ctx_t ctx;
            auto out = U::forward(ctx, std::forward<Args>(args)...);

            auto captured_vars = std::make_tuple(std::forward<Args>(args)...);

            out->grad_fn = [ctx, captured_vars, out](auto grad) mutable
            {
                U::backward(ctx, grad);

                // 這裡應該要自動handle 所有變量的backward.
                // std::apply([](auto... vars)
                //            { (void(std::initializer_list<int>{
                //                  (call_backward_if_tensor_ptr(vars), 0)...})); }, captured_vars);
            };
            return out;
        }
    };

    template <typename T>
    void call_backward_if_tensor_ptr(T var)
    {
        if constexpr (std::is_same_v<std::decay_t<T>, FloatTensorPtr> ||
                      std::is_same_v<std::decay_t<T>, CudaFloatTensorPtr>)
        {
            if (var->requires_grad && var->grad_fn != nullptr) // Segmentation fault here
            {
                var->backward(var->get_grad());
            }
        }
    }

    class Add : public FnBase<Add>
    {
    public:
        static FloatTensorPtr forward(ctx_t &ctx, FloatTensorPtr x, FloatTensorPtr y)
        {
            auto output = std::make_shared<FloatTensor>(x->get_shape());
            auto outdata = output->get_data().get();
            auto xdata = x->get_data().get();
            auto ydata = y->get_data().get();

            for (int i = 0; i < x->get_numel(); i++)
            {
                outdata[i] = xdata[i] + ydata[i];
            }

            ctx["x"] = x;
            ctx["y"] = y;

            return output;
        }

        static void backward(ctx_t &ctx, FloatTensorPtr grad)
        {
            auto x = std::get<FloatTensorPtr>(ctx["x"]);
            auto y = std::get<FloatTensorPtr>(ctx["y"]);

            // x->print();
            // x->fill_(1.0);

            if (grad == nullptr)
            {
                grad = fisea::FloatTensor::create(x->get_shape());
                grad->fill_(1);
            }

            if (x->requires_grad)
            {
                x->set_grad(grad);
            }

            if (y->requires_grad)
            {
                y->set_grad(grad);
            }

            return;
        }
    };

} // namespace fisea