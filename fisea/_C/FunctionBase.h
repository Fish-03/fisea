
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

#include <tuple>
#include <type_traits>

#include "type.h"
#include "FloatTensor.h"
namespace fisea
{
    template <typename T, typename = void>
    struct has_grad_fn : std::false_type
    {
    };
    
    template <typename T, typename = void>
    struct is_leaf : std::false_type
    {
    };

    template <typename T>
    struct is_leaf<T, std::void_t<decltype(std::declval<T>()->is_leaf)>> : std::true_type
    {
    };

    template <typename T>
    struct has_grad_fn<T, std::void_t<
                              decltype(std::declval<T>()->grad_fn(std::declval<decltype(std::declval<T>()->get_grad())>()))>> : std::true_type
    {
    };

    template <typename argTuple, typename gradTuple, size_t Index = 0>
    std::enable_if_t<Index == std::tuple_size_v<argTuple>, void>
    call_grad_fn_if_exists(const argTuple &t, const gradTuple &g, bool retain_graph, bool create_graph) {}

    template <typename argTuple, typename gradTuple, size_t Index = 0>
        std::enable_if_t < Index<std::tuple_size_v<argTuple>, void>
                           call_grad_fn_if_exists(const argTuple &t, const gradTuple &g, bool retain_graph, bool create_graph)
    {
        using T = std::tuple_element_t<Index, argTuple>;
        T obj = std::get<Index>(t);
        auto grad = std::get<Index>(g);

        // 如果对象存在 grad_fn 并且不是 leaf 节点，调用 grad_fn，否则调用 set_grad
        if (obj->grad_fn)
        {
            obj->grad_fn(grad, retain_graph, create_graph); // 否则调用 grad_fn
        }

        if (obj->is_leaf)
        {
            obj->set_grad(grad); // 如果是 leaf，调用 set_grad
        }

        call_grad_fn_if_exists<argTuple, gradTuple, Index + 1>(t, g,  retain_graph,  create_graph);
    }

    template <typename T>
    class FnBase
    {

    public:
        template <typename U = T, typename... Args>
        static auto apply(Args... args) -> decltype(U::forward(std::declval<ctx_t &>(), std::forward<Args>(args)...))
        {
            ctx_t ctx;
            auto captured_vars = std::make_tuple(args...);

            // 然后再转发参数，可能会移动参数的所有权
            auto out = U::forward(ctx, std::forward<Args>(args)...);

            out->grad_fn = [ctx, captured_vars, out](auto grad, bool retain_graph = false, bool create_graph = false) mutable
            {
                auto gradTuple = U::backward(ctx, grad);
                call_grad_fn_if_exists(captured_vars, gradTuple, retain_graph, create_graph);
                if (!retain_graph)
                {
                    out->grad_fn = nullptr;
                }
            };
            return out;
        }
    };

    class Add : public FnBase<Add>
    {
    public:
        static FloatTensorPtr forward(ctx_t &ctx, FloatTensorPtr x, FloatTensorPtr y);
        static std::tuple<FloatTensorPtr, FloatTensorPtr> backward(ctx_t &ctx, FloatTensorPtr grad);
    };

} // namespace fisea