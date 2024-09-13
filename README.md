# fisea

Develop Manual

As you can see, the `fisea` file store all the `.h`, `.cpp`, and `.cu`, mixed with python codes. Our target is to provide some `C++` functions or `CU` functions to python. Giving the package user a torch-like experience in developing new things.

## TODO
- In fisea.functional store all `simple function acting on one element`. Functions with names ended with `_`, for example `sin_`, `cos_`, ..., are `inplace function`. All(most) functions will have one inplace version and one creating-new-copy version.

- In fisea.tensor store the `tensor` API for the user. We need to implement a lot of build-in functions that directly call the `C++` `CU` functions written in fisea.functional

- pip support, cuda support

- conda support

- since there is no documentation or type hint for `c++` functions, so we need to write some `.pyi` files, similar to `.h` files, containing all of the necessary documentations.

# Test it!
```python
from fisea.functional import mul, div

a = 10
b = 5

print(mul(a, b))
```