from ._cpp_extension import add, sub, mul, div
try:
    from ._cpp_extension import cuda_test
except ImportError:
    pass