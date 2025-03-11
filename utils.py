import os
import numpy as np

CENTS_ROUND=3
RATIO_ROUND=6

CENTS_LOOKUP={}
def ratio_to_cents(ratio):
    if ratio in CENTS_LOOKUP:
        return CENTS_LOOKUP[ratio]
    CENTS_LOOKUP[ratio] = np.round(np.log2(ratio) * 1200, CENTS_ROUND) 
    return CENTS_LOOKUP[ratio]

def nd_basis_to_cents(nd_periods):
    to_cents = np.vectorize(ratio_to_cents)
    cents = to_cents(nd_periods[:,1:] / nd_periods[:,:-1])
    return cents.squeeze()

def cents_to_ratio(cents):
    return np.exp2(cents / 1200.0)

def get_cf(num, maxdepth=20, round0thresh=1e-5):
    n = num
    cf = []
    for i in range(maxdepth):
        cf.append(int(n))
        n -= cf[i]

        if (n > round0thresh):
            n = 1 / n
        else:
            break

    return cf

def get_convergent(cf, depth=-1):
    nums = cf
    while (len(nums) > 0 and nums[-1] == 0):
        nums = nums[:-1]

    size = len(nums)
    if (size == 0):
        return (0, 0)

    if (depth >= size or depth < 0):
        depth = size - 1

    (num, den) = (1, nums[depth])
    
    for d in range(depth, 0, -1):
        factor = nums[d-1]
        num += factor * den
        (num, den) = (den, num)
        
    return (den, num)

def get_bcf(num, maxdepth=20, round0thresh=1e-7):
    n = num
    bcf = []
    for i in range(maxdepth):
        bcf.append(int(n) + 1)
        n = bcf[i] - n
        if (n > round0thresh):
            n = 1 / n
        else:
            break

    return bcf

def get_bcf_until(num, numTwos=64, numTest=None, max_depth=1024, round0thresh=1e-7):
    n = num
    bcf = []
    i = 0
    count_2s = 0

    if numTest is None:
        numTest = lambda x: False

    for i in range(max_depth):
        bcf.append(int(n) + 1)        
        if bcf[i] == 2:
            count_2s += 1
        else:
            count_2s = 0

        n = bcf[i] - n

        if n <= round0thresh:
            break

        if count_2s >= numTwos:
            break

        if numTest(bcf[i]):
            break

        n = 1 / (n)

    return bcf

def getUniqueFilename(filename, type):
    out = os.path.join(f'{filename}.{type}')
    if not os.path.exists(os.path.join(out)):
        return out
    max_iter = 99
    Path=lambda f: os.path.join(f"{filename}_{f}.{type}")
    f=0
    for f in range(max_iter):
        f+=1
        out=Path(f)
        while os.path.exists(out):
            f+=1
            out=Path(f)
        break
    return out

# convert int matrix with shape (N, 2) to a harmonic segment
def normalize_ratios(ratio_list):
    if ratio_list.shape[0] == 1:
        return ratio_list

    # normalize into a harmonic segment
    for k in range(ratio_list.shape[0] - 1):
        r1 = ratio_list[k,:]
        r2 = ratio_list[k+1,:]

        m = np.lcm(r1[1], r2[0])
        
        r1_factor = m // r1[1]
        r2_factor = m // r2[0]

        for j in range(k+1):
            ratio_list[j,:] *= r1_factor
        ratio_list[k+1,:] *= r2_factor

    return np.asarray([ ratio_list[0,0], *ratio_list[:, 1]], dtype=np.uint64)

# convert to close harmonic chord
# expects list of ratios as consecutive tones from root
# TODO: option for absolute ratios?
# returns ( harmonics(1,len(decimal_list) + 1 ), cents_errors )
def approximate_ratio(decimal_list, int_limit=1024, max_cents_tolerance=0.6, passError=False):

    shape = (len(decimal_list), 2)
    ratios = np.ndarray(shape, dtype=np.uint32)
    error = 0
    errors = np.zeros(shape[0])

    # get approximated dyads
    for k in range(shape[0]):
        ratio = decimal_list[k]
        cf = get_cf(ratio)

        cents = ratio_to_cents(ratio)

        [n, d] = get_convergent(cf,0)

        conv_cents = ratio_to_cents(n / d)
        cents_dist = abs(cents - conv_cents)

        i = 1
        while cents_dist > max_cents_tolerance:
            conv = get_convergent(cf, i)
            (cn, cd) = conv
            if cn <= int_limit and cd <= int_limit:
                (n, d) = conv
                conv_cents = ratio_to_cents(cn/cd)
                cents_dist = abs(cents - conv_cents)
            else:
                break
            i+=1

        ratios[k, :] = [d, n]
        error += cents_dist
        errors[k] = cents_dist

    approximation = normalize_ratios(ratios)
    if passError:
        return (approximation, errors)
    return approximation

def get_edo_approx(cents, cents_threshold, max_edo=313, max_depth=100, round_threshold=0.0000001):
    r_2 = round(cents / 1200, 6)
    cf = get_cf(r_2, max_depth, round_threshold)

    edo = 1
    degree = 1
    error = 0

    for i in range(1, len(cf)):
        (next_degree, next_edo) = get_convergent(cf, i)

        if edo > max_edo:
            break
        
        edo = next_edo
        degree = next_degree
        
        approx = 1200 / edo * degree
        error = cents - approx

        if abs(error) <= cents_threshold:
            break

    return (edo, degree, error)

# EASING FUNCTIONS

def EaseIoSlope(t, a=2):
    d_base = 1 + 10 ** (a * 0.5)
    return (1.0 / (1 + 10 ** (a * (0.5 - t))) - 1 / (d_base)) / (1.0 - 2 / d_base)

if __name__ == '__main__':
    import random
    import sys

    N = 10
    if len(sys.argv) > 1:
        N = int(sys.argv[1])

    # for i in range(100):
    #     r = random.Random()
    #     ratio = r.random() + 1
    #     print(f"Continued fraction of {ratio}:\t{get_cf(ratio)}")

    print(normalize_ratios(np.asarray([[4,5],[10,11],[2,3]])))
    print(normalize_ratios(np.asarray([[4,5],[10,11]])))
    print(normalize_ratios(np.asarray([[10,11]])))
    

