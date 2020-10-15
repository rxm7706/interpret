# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, overflowcheck=False, cdivision=True, infer_types=True
# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp

cimport cython
import numpy as np
cimport numpy as np

ctypedef fused num:
    double
    long long

cpdef int sum(int a, int b):
    return a + b