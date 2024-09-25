# fisea

Develop Manual

As you can see, the `fisea` file store all the `.h`, `.cpp`, and `.cu`, mixed with python codes. Our target is to provide some `C++` functions or `CU` functions to python. Giving the package user a torch-like experience in developing new things.

## TODO
- In fisea.functional store all `simple function acting on one element`. Functions with names ended with `_`, for example `sin_`, `cos_`, ..., are `inplace function`. All(most) functions will have one inplace version and one creating-new-copy version.

- In fisea.tensor store the `tensor` API for the user. We need to implement a lot of build-in functions that directly call the `C++` `CU` functions written in fisea.functional

- conda support

- since there is no documentation or type hint for `c++` functions, so we need to write some `.pyi` files, similar to `.h` files, containing all of the necessary documentations.

# Test it!
```python
from fisea.functional import mul, div

a = 10
b = 5

print(mul(a, b))
```

## Devlopment Log (2024-09-24)

Tensor類的cpu版本完成了很多, 注意一下 如果函數以`_`開頭, 那麼這個函數是不想被用戶所調用的, 因此基本上都定義在 private 內. 而若函數以`_`結尾, 那麼這個函數是 in-place 函數. 比如 將`Tensor`由`fisea::Dtype::FLOAT` 轉換到 `fisea::Dtpye::INT` 的函數, 對於兩個版本: `Tensor::to_int` 和 `Tensor::to_int_`. 兩個版本各有`cpu`和`cuda`處理的內部函數. 因此代碼量極大.

現在還差`Tensor.cu`內大量的內容, 還有`functional/basickr.cu`內有很多kernel函數. 我的想法是Tensor內的其他更為複雜的函數, 比如`Tensor::add`, 全部都通過 調用`functional`內的basic function 來實現, 比如
```cpp
Tensor add(Tensor a, Tensor b) {
    Tensor c = Tensor(a.size(), a.dtype());
    // SOME KERNEL FUNCTION HERE
}

Tensor Tensor::add_(Tensor b) {
    // SOME KERNEL FUNCTION HERE
    return *this;
}

Tensor Tensor::add(Tensor b) {
    return add(c, b);
}
```
還有很多代碼要寫, 魚魚加油.

正確的結構應該是 `.h`可以對應`.cu` 和`.cpp`, 而`.cuh`只對於`.cu`. 而且`.h`和`.cpp`是不能include `cuh`的, 這也要注意.  但是相返則是可以的.

## Devlopment Log (2024-09-25)
要讓 .cpp 文件使用 .cuh 作為頭的話 (使用裡面的函數)
- .cuh 文件不能定義有關kernel的函數，比如 `__global__` 這種。
- 模板函數需要在.cu文件中先實例化。

現在windows也能編譯出這段代碼了，而且生成了_C_EXE來進行debug，具體可以在main.cpp裡進行。