"""

  Generate basis sets for harmonic entropy 

"""

import ctypes
import os
import numpy as np
from math import gcd
import numpy as np
import utils
from primes import *

try:
    _c_farey_path=os.path.abspath('./c_farey/farey.so')
    __utils = ctypes.CDLL(_c_farey_path)

    __intp = ctypes.POINTER(ctypes.c_int)

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

def farey_set_to_basis(ndSet, harmonic_limit=2):
    size = ndSet.shape[0]

    octaves = np.ceil(np.log2(harmonic_limit))
    period_shape = (int(size * np.exp2(octaves-1) - (harmonic_limit > 2)), 2)

    basis = np.zeros(period_shape, dtype=np.uint) # overestimate
    basis[:size, 0] = ndSet[:, 1]
    basis[:size, 1] = ndSet[:, 0] + ndSet[:, 1]

    i = 1
    p = size
    row = ndSet[-1,:]
    limit = np.asarray([ 1, harmonic_limit ])

    while p < period_shape[0] and not np.array_equal(row, limit):
        if np.mod(basis[i, 0], 2) == 0:
            row = [ basis[i, 0] // 2, basis[i, 1] ]
        else:
            row = [ basis[i, 0], basis[i, 1] * 2 ]

        basis[p,:] = row
        i += 1; p += 1

    return basis[:p,:]

def get_farey_sequence_basis(N, harmonic_limit=2):
    return farey_set_to_basis(farey(N), harmonic_limit)


def get_triplet_basis(N, harmonic_limit=2, product_limit=27_000_000):
    harmonic = int(max(harmonic_limit, 2))

    # generate all triples a:b:c within the period
    triplets = []
    for i in range(1, N):
        for j in range(i, harmonic*i+1):
            for k in range(j, harmonic*i+1):
                if i*j*k < product_limit and gcd(i,j,k) == 1:
                    triplets.append([i,j,k])

    return np.asarray(triplets)

def nd_basis_to_cents(nd_periods):
    to_cents = np.vectorize(utils.ratio_to_cents)
    cents = to_cents(nd_periods[:,1:] / nd_periods[:,:-1])
    return cents.squeeze()

def create_mask_from_lambdas(basis_set, *funcs):
    def composite(*args):
        result = funcs[0](args[0])
        for f in funcs[1:]:
            result &= f(basis_set)
        return result
    return composite(basis_set)

# Test for integer set only including primes from this set
def create_prime_group_test(primes):
    def check(n):
        prime_list = get_prime_list(n)
        for p in prime_list:
            if p not in primes:
                return False
        return True
    return np.apply_along_axis(lambda row: np.any(np.vectorize(check)(row)), 1, basis_set)

# Test for integer set only including primes from this set
def create_prime_limit_test(primes):
    def test(basis_set, primes):
        limit = primes[-1]
        def check(n):
            prime_list = get_prime_list(n)
            for p in prime_list:
                if p > limit:
                    return False
            return True
        return np.apply_along_axis(lambda row: np.all(np.vectorize(check)(row)), 1, basis_set)
    return lambda basis: test(basis, primes)

# Test for integer set containing all of these primes
def create_exact_prime_test(primes):
    def test(basis_set, primes):
        def check(n):
            for p in primes:
                if p % n == 0:
                    return True
            return False
        return np.apply_along_axis(lambda row: np.all(np.vectorize(check)(row)), 1, basis_set)
    return lambda basis: test(basis, primes)

# Test for integer set not including any of these primes
def create_exclusive_prime_test(primes):
    def test(basis_set, primes):
        def check(n):
            for p in primes:
                return n % p > 0
        return np.apply_along_axis(lambda row: np.all(np.vectorize(check)(row)), 1, basis_set)
    return lambda basis: test(basis, primes)

def create_is_prime_test():
    def test(basis_set):
        return np.apply_along_axis(lambda row: np.all(np.vectorize(is_prime)(row)), 1, basis_set)
    return test

if __name__ == '__main__':
    N = 10
    harmonic = 2
    basis_set = get_farey_sequence_basis(N, harmonic)
    print(f"N: {N}, set size: {len(basis_set)}")
    basis_cents = nd_basis_to_cents(basis_set)

    triplet_set = get_triplet_basis(N, harmonic)
    print(f'\ttriplet size: {len(triplet_set)}')
    triplet_cennts = nd_basis_to_cents(triplet_set)

    print("Done")
