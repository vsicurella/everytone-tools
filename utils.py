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
    cf = [] # the continued fraction
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

# convert to close harmonic chord
# expects list of ratios as consecutive tones from root
# TODO: option for absolute ratios?
# returns ( harmonics(1,len(decimal_list) + 1 ), cents_errors )
def approximate_ratio(decimal_list, int_limit=1024, max_cents_tolerance=0.6):

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

    if shape[0] == 1:
        approximation = np.asarray([ d, n ], dtype=np.uint64)
    
    else:
        # normalize into a harmonic segment
        for k in range(shape[0] - 1):
            r1 = ratios[k,:]
            r2 = ratios[k+1,:]

            m = np.lcm(r1[1], r2[0])
            
            r1_factor = m // r1[1]
            r2_factor = m // r2[0]

            for j in range(k+1):
                ratios[j,:] *= r1_factor
            ratios[k+1,:] *= r2_factor

        approximation = np.asarray([ ratios[0,0], *ratios[:, 1]], dtype=np.uint64)

    return (approximation, errors)

if __name__ == '__main__':
    import random
    import sys

    N = 10
    if len(sys.argv) > 1:
        N = int(sys.argv[1])

    for i in range(100):
        r = random.Random()
        ratio = r.random() + 1
        print(f"Continued fraction of {ratio}:\t{get_cf(ratio)}")

