"""

  Generate basis sets for harmonic entropy 

"""

import numpy as np
from math import gcd
from primes import *
from farey import * 
# from utils import *

_RATIO_DB_ = None
_USE_RATIO_DB_ = True
try:
    _RATIO_DB_ = __import__("ratio-db")
except:
    _USE_RATIO_DB_ = False
    print("Unable to import ratio-db.py, perhaps due to a missing dependency. Using fallback code.")


# ndSet, basis: [ denominator, numerator ]
def farey_set_to_basis(ndSet, harmonic_limit=2):
    size = ndSet.shape[0] - 1

    octaves = np.ceil(np.log2(harmonic_limit))
    period_shape = (int((size+1) * np.exp2(octaves-1) - (harmonic_limit > 2)), 2)

    basis = np.zeros(period_shape, dtype=np.uint) # overestimate
    basis[:size, 0] = ndSet[:size, 1]
    basis[:size, 1] = ndSet[:size, 0] + ndSet[:size, 1]

    i = 0
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

def get_triplet_basis(N, harmonic_limit=2, product_limit=27_000_000, start_harmonic=1):
    if harmonic_limit < 2:
        raise Exception("harmonic_limit cannot be less than 2")
    harmonic = int(harmonic_limit)

    top_limit = N * harmonic
    if start_harmonic < 1 or start_harmonic >= top_limit:
        raise Exception(f"start_harmonic should be 1 or below {top_limit}")
    start = int(start_harmonic)

    # generate all triples a:b:c within the period
    triplets = []
    for i in range(start, N):
        for j in range(i, harmonic*i+1):
            for k in range(j, harmonic*i+1):
                if i*j*k < product_limit and gcd(i,j,k) == 1:
                    triplets.append([i,j,k])

    return np.asarray(triplets)

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

class BasisParams:
    def __init__(self, complexity_limit, integer_limit=None, he3=False, noDyads=False):
         self.complexity_limit = complexity_limit
         self.integer_limit = integer_limit
         self.he3 = he3
         self.noDyads = noDyads
class PrimeLimitBasisParams(BasisParams):
    def __init__(self, limit, harmonic_limit, complexity_limit, integer_limit=None, he3=False, noDyads=False):
        super().__init__(complexity_limit, integer_limit, he3, noDyads)
        self.limit = limit
        self.harmonic_limit = harmonic_limit

class BasisInterface:
    ratioDb = None

    __dummy_find__=lambda tagsString,matchString,sortString,offset,queryLimit: -1
    find_harmonics = __dummy_find__
    find_dyads = __dummy_find__
    find_triads = __dummy_find__
    
    get_farey_sequence_basis = lambda self, N, harmonic_limit: get_farey_sequence_basis(N, harmonic_limit)
    get_prime_basis = lambda params: [] # TODO

    _create_prime_limit_iterator_ = lambda prime, limit: iter(range(0)) # TODO

    def _get_farey_sequence_basis_db_(N, harmonic_limit):
        basis_set = BasisInterface.find_dyads('values', f'1200 >= any(cents) and prime_limit <= ({N*2-1}) and (numerator<={N} or denominator<={N}))', 'integer_limit DESC')
        # todo figure this query out
        return farey_set_to_basis(basis_set, harmonic_limit)
    
    def _get_harmonic_segment_db_(self, start=8, end=16, prime_limit=13, sortString=None, limit=0):
        harmonics = self.find_harmonics(tagString="harmonic", matchString=f'prime_limit={prime_limit} and harmonic >= {start} and harmonic <= {end}', sortString=sortString, limit=limit)
        return iter([ h[0] for h in harmonics ])
    
    def _create_harmonic_segment_iterator_db_(self, start=8, end=16, prime_limit=13):
        harmonics = self._get_harmonic_segment_db_(start, end, prime_limit)
        return iter(harmonics)

    def _get_prime_basis_db_(self, params:PrimeLimitBasisParams):
            prime_match = f'prime_limit<={params.limit}'
            if params.complexity_limit:
                prime_match += f' and complexity_limit<={params.complexity_limit}'
            if params.integer_limit:
                prime_match += f' and integer_limit<={params.integer_limit}'
            prime_match += f' and (SELECT MAX(d) FROM unnest(decimals) AS d) <= {params.harmonic_limit}'
            if params.he3:
                triad_match = prime_match
                if params.noDyads:
                    triad_match += f' and (root < mediant and mediant < dominant)'
                # triad_match += f' and '
                return self.find_triads(matchString=triad_match, sortString='integer_limit ASC, root ASC')
            return self.find_dyads(matchString=prime_match, sortString='any(cents) ASC')

    def __init__(self):
        if _USE_RATIO_DB_:
            BasisInterface.ratioDb          = _RATIO_DB_.RatioDb()
            self.find_harmonics             = BasisInterface.ratioDb.find_harmonics
            self.find_dyads                 = BasisInterface.ratioDb.find_dyads
            self.find_triads                = BasisInterface.ratioDb.find_triads
            self.get_farey_sequence_basis   = self._get_farey_sequence_basis_db_
            self.get_prime_basis            = self._get_prime_basis_db_
            self._create_harmonic_segment_iterator_db_ = self._create_harmonic_segment_iterator_db_

class Basis(BasisInterface):

    def __init__(self,**kwArgs):
        super().__init__()

        self.he3 = False if "he3" not in kwArgs else kwArgs["he3"]
        self.noDyads = False if "noDyads" not in kwArgs else kwArgs["noDyads"]

        self.params = None if "params" not in kwArgs else kwArgs["params"]

    def setToPrimeLimit(self, limit, harmonic_limit=2, complexity_limit=None, integer_limit=None):
        # if complexity_limit:
        #     self.prime_query.complexity_limit = complexity_limit
        # else:
            # end_harm = limit * harmonic_limit
            # [[adj_harm]] = self.find_harmonics("harmonic", f"prime_limit={params.limit} and harmonic < {end_harm}", 'harmonic DESC', queryLimit=1)
            # c_limit = (end_harm) ** (3 - (1 if params.he3 else 2)) * adj_harm
            # c_limit = end_harm * harmonic_limit * (harmonic_limit if self.he3 else 1)
            # c_limit = end_harm * harmonic_limit

        if integer_limit is None:
            integer_limit = limit * harmonic_limit ** 2

        self.prime_query = PrimeLimitBasisParams(limit, harmonic_limit, complexity_limit, integer_limit, self.he3, self.noDyads)
        if self.prime_query == self.params:
            return

        basis = self.get_prime_basis(self.prime_query)
        self._set_basis_(basis) # move to interface?
        self.params = self.prime_query

    # def setToPrimodalStructure(self, prime, start=2, end=4):
    #     self.prime_query = PrimeLimitBasisParams(prime, end, None, prime*end, self.he3, self.noDyads)
    #     self.basis = self.get_prime_basis(self.prime_query)
    #     self.params = self.prime_query

    def setToIntegerLimit(self, int_limit, harmonic_limit):
        #todo params
        if self.he3:
            self.basis = self.find_triads()
        else:
            return get_farey_sequence_basis(int_limit, harmonic_limit)
        
    def _set_basis_(self, basis):
        self.basis = basis
        self.values     = np.asarray([ data[1] for data in basis ])
        self.labels     = np.asarray([ data[2] for data in basis ])
        self.decimals   = np.asarray([ data[5] for data in basis ])
        self.cents      = np.asarray([ data[7] for data in basis ])
        self.he_weights = np.asarray([ data[17] for data in basis  ])
        
    def getSet(self):
        return self.basis
        
    def getValues(self):
        return self.values
    
    def getLabels(self):
        return self.labels
    
    def getDecimals(self):
        return self.decimals
    
    def getCents(self):
        return self.cents
    
    def getWeights(self):
        return self.he_weights

if __name__ == '__main__':
    
    N = 5
    harmonic = 4
    basis_set = get_farey_sequence_basis(N, harmonic)
    duplicateCheck = {}
    i = 0
    for dyad in basis_set:
        dyad = f'{dyad[1]}_{dyad[0]}'
        if dyad in duplicateCheck:
            raise Exception(f"Duplicate dyad! At index {i}")
        i+=1
        duplicateCheck[dyad] = True
    print("successful dyad test")

    # print(f"N: {N}, set size: {len(basis_set)}")
    # basis_cents = nd_basis_to_cents(basis_set)

    # triplet_set = get_triplet_basis(N, harmonic)
    # print(f'\ttriplet size: {len(triplet_set)}')
    # triplet_cennts = nd_basis_to_cents(triplet_set)

    basis = Basis(he3=True)
    basis.setToPrimeLimit(5, 4)

    print("Done")
