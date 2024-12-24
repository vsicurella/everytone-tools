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

if __name__ == '__main__':
    import random
    r = random.Random()
    ratio = r.random() + 1
    print(f"Continued fraction of: {ratio}")
    print(get_cf(ratio))

