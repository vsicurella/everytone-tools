import ctypes
import os
import numpy as np

try:
    __utils = ctypes.CDLL(os.path.abspath('./pyc_utils/utils.so'))

    __intp = ctypes.POINTER(ctypes.c_int)

    # _farey_test = __utils.farey_test
    # _farey_test.argtypes = [ctypes.c_int]

    _farey_len = __utils.farey_len
    _farey_len.argtypes = [ctypes.c_int]
    _farey_len.argtypes = [ctypes.c_ulonglong]

    _farey_c = __utils.farey
    _farey_c.argtypes = [ctypes.c_int, ctypes.c_ulonglong, __intp]

    def _farey(n):
        size = farey_len(n)
        seq = np.zeros((size * 2), dtype=np.int32)
        pseq = seq.ctypes.data_as(__intp)
        _farey_c(n, size, pseq)
        return seq.reshape((size, 2))

except:
    print("Error, unable to load utils binary, falling back to numpy version")

    def _farey_len(n):
        len = n * (n + 3) // 2
        p = 2
        q = 0
        while p <= n:
            q = n // (n // p) + 1
            len -= _farey_len(n // p) * (q - p)
            p = q
        return len
    
    def _farey(n):
        f1 = (1, 0)
        f2 = (n, 1)
        
        l = _farey_len(n)
        nd_interleaved = np.zeros((l, 2), dtype=np.int64)
        nd_interleaved[0, :] = f1[::-1]
        nd_interleaved[1, :] = f2[::-1]

        i = 2
        while (f2[0] > 1):
            k = (n + f1[0]) // f2[0]

            t = f1
            f1 = f2
            f2 = (f2[0] * k - t[0], f2[1] * k - t[1])

            nd_interleaved[i, :] = f2[::-1]
            i+=1

        return nd_interleaved

def farey_len(n):
    return _farey_len(n)

def farey(n):
    return _farey(n)


def get_cf(num, maxdepth=20, round0thresh=1e-5):
    n = num
    cf = [] # the continued fraction
    for i in range(maxdepth):
        cf.append(int(n))
        n -= cf[i]

        if (n > round0thresh):
            n = 1 / n
        else:
            break

    return cf

if __name__ == '__main__':
    size = 50
    print("Farey sequence N="+str(size))
    print("\tLength: " + str(farey_len(size)))

    # _farey_test(size)
    
    seq = farey(size)
    print(seq)
    print("Success")

