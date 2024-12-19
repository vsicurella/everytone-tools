import ctypes
import os
import numpy as np

__utils = ctypes.CDLL(os.path.abspath('pyc_utils/utils.so'))

__intp = ctypes.POINTER(ctypes.c_int)

__farey_c = __utils.farey
__farey_c.argtypes = [ctypes.c_int, ctypes.c_ulonglong, __intp]

__farey_test_c = __utils.farey_test
__farey_test_c.argtypes = [ctypes.c_int]

__farey_len_c = __utils.farey_len
__farey_len_c.argtypes = [ctypes.c_int]
__farey_len_c.argtypes = [ctypes.c_ulonglong]

def farey_len(n):
    return __farey_len_c(n)

def farey(n):
    size = farey_len(n)
    seq = np.zeros((size * 2), dtype=np.int32)
    pseq = seq.ctypes.data_as(__intp)
    __farey_c(n, size, pseq)
    return seq.reshape((size, 2))

if __name__ == '__main__':
    size = 5
    print("Farey sequence N="+str(size))
    print("\tLength: " + str(farey_len(size)))

    __farey_test_c(size)
    
    seq = farey(size)

    print("Success")

