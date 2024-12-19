"""

  https://en.xen.wiki/w/Harmonic_entropy

"""


import utils
import numpy as np

import matplotlib.pyplot as plt
# from scipy.signal import fftconvolve

import os

CENTS_ROUND=3
RATIO_ROUND=6

def farey_set_to_basis(ndSet, octaves=1):
    size = ndSet.shape[0]
    period_shape = (size * octaves - 1, 2)

    basis = np.zeros(period_shape, dtype=np.uint)
    basis[:size, 0] = ndSet[:, 0] + ndSet[:, 1]
    basis[:size, 1] = ndSet[:, 1]

    i = 0
    p = size
    for i in range(1, period_shape[0] - size + 1):
        if np.mod(basis[i, 1], 2) == 0:
            basis[p,:] = [ basis[i, 0], basis[i, 1] // 2 ]
        else:
            basis[p,:] = [ basis[i, 0] * 2, basis[i, 1] ]
        p += 1
    return basis

def ratio_to_cents(ratio):
    return np.round(np.log2(ratio) * 1200, CENTS_ROUND)

def show_plot(data, title="Plot", xlabel="X", ylabel="Y", **kwArgs):
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if kwArgs:
        for kw in kwArgs:
            if kw == "annotate":
                txtpos = (0, max(data))
                plt.annotate(kwArgs[kw], txtpos)
            elif kw == "xticks":
                plt.xticks(kwArgs[kw])
            elif kw.startswith("plot"):
                plt.plot(kwArgs[kw])

    plt.show()

def is_odd(integer):
    return integer & 1

def ratio_has_evens(ndRatio):
    return not (ndRatio[0] & ndRatio[1] & 1)


class HarmonicEntropy:
    s = None
    N = None
    res = None
    limit = None
    a = None

    basis_filter = None

    # state
    x_axis = None

    num_periods = None
    basis_set = None
    basis_periods = None
    basis_ratios = None
    basis_cents = None

    x_length = None
    basis_length = None

    Q = None
    Qsum = None
    P = None
    Entropy = None

    # pre-computes
    i_ss2 = None
    ssqrt2pi = None

    def __init__(self, s=17, N=100, res=1, limit=2400, a=1):
        self.update(s, N, res, limit, a)

    def suffix(self):
        return "{}s_{}N_{}c_{}max".format(self.s, self.N, self.res, self.limit)

    def update(self, s=None, N=None, res=None, limit=None, a=None):
        if s is not None:
            self.s = s
        if N is not None:
            self.N = N
        if res is not None:
            self.res = res
        if limit is not None:
            self.limit = limit
        if a is not None:
            self.a = a

        self.x_axis = np.asarray(range(0, round(limit+res), res))
        self.x_length = len(self.x_axis)

        self.num_periods = int(np.ceil(self.limit / 1200))

        self.i_ss2 = 1 / (self.s**2 * 2);
        self.ssqrt2pi = self.s * 2.50662827463;

    def prepareBasis(self, basis=None):
        if basis:
            self.basis_set = basis["set"]
            self.basis_periods = basis["periods"]
            self.basis_ratios = basis["ratios"]
            self.basis_cents = basis["cents"]
            self.basis_length = len(self.basis_cents)
            return
        
        saveData=True
        file = os.path.join(os.path.dirname(__file__), "he_data", "farey{}.npy".format(self.N))
        if os.path.exists(file):
            basis = np.load(file)
        else:
            print("Calculating rationals...")
            basis = utils.farey(self.N)
            if saveData:
                np.save(file, basis)

        self.updateBasisSet(basis)

    def updateBasisSet(self, newBasis):
        self.basis_set      = newBasis
        self.basis_periods  = farey_set_to_basis(self.basis_set, self.num_periods)

        self.basis_ratios   = np.round(self.basis_periods[:, 0] / self.basis_periods[:, 1], RATIO_ROUND)
        self.basis_cents    = np.vectorize(ratio_to_cents)(self.basis_ratios)
        self.basis_length   = len(self.basis_cents)

    def setBasisMask(self, mask):
        basis_n = np.extract(mask, self.basis_periods[:,0])
        basis_d = np.extract(mask, self.basis_periods[:,1])
        self.basis_periods  = np.column_stack((basis_n, basis_d))
        self.basis_ratios   = np.extract(mask, self.basis_ratios)
        self.basis_cents    = np.extract(mask, self.basis_cents)
        self.basis_length   = len(self.basis_cents)

    def setOddBasis(self):
        mask = self.basis_periods[:, 0] & self.basis_periods[:, 1] & 1
        self.setBasisMask(mask)
        
    def prepareWeights(self, weights=None):
        if weights:
            return
        
        print("Calculating weights...")

        products = self.basis_periods[:,0] * self.basis_periods[:,1]
        weights = np.reciprocal(self.ssqrt2pi * np.sqrt(products))
        
        diffs = np.asarray([ self.basis_cents[j] - self.x_axis for j in range(self.basis_length)])
        self.Q = np.asarray([ np.exp(-np.square(diffs[j,:]) * self.i_ss2) * weights[j] for j in range(self.basis_length) ])
        self.Qsum = self.Q.sum(axis=0)

    def prepareProbabilities(self, probabilities=None):
        if probabilities:
            return
        
        print("Calculating probabilities...")

        sigma = 1e-12

        self.P = np.zeros((self.basis_length, self.x_length))

        for c in range(self.x_length):
            self.P[:, c] = self.Q[:, c] / self.Qsum[c] + sigma
        
    def prepareEntropy(self, entropy=None):
        if entropy:
            return
        
        print("Calculating entropy...")
    
        self.Entropy = np.zeros(self.x_length)
        for c in range(self.x_length):
            self.Entropy[c] = -sum(self.P[:,c] * np.log(self.P[:,c]))

    def convolveHRE(self):
        base_weighted = 1 / np.sqrt(self.basis_periods[:,0] * self.basis_periods[:,1])
        base_weighted_alpha = np.power(base_weighted, self.a)

        # turn basis cents into a sum of delta functions
        basis_index = np.rint(self.basis_cents / self.res).astype(np.uint32)

        K = np.zeros(self.x_length)
        Ka =  np.zeros(self.x_length)
        for ji in range(self.basis_length):
            di = self.x_length - basis_index[ji] - 1
            K[di] += base_weighted[ji]
            Ka[di] += base_weighted_alpha[ji]

        # show_plot(K, "K")
        # return

        S = np.exp(-np.square(self.x_axis - self.x_length//2) * self.i_ss2) / self.ssqrt2pi
        # show_plot(S, "S")
        # return
    
        psi = np.convolve(K, S, 'same')

        if self.a == 1:
            self.Entropy = psi / np.log(psi**self.a)
            return True

        pa = np.convolve(Ka, np.power(S, self.a), 'same')

        self.Entropy = np.log(pa / psi ** self.a) / (1 - self.a)

        return True

    def calculate(self):
        self.prepareBasis()
        if self.convolveHRE():
            return
        # self.prepareWeights()
        # self.prepareProbabilities()
        # self.prepareEntropy()

    def plot(self):
        diff = np.diff(self.Entropy)
        # diff2 = np.diff(diff)
        ticks = []
        for i in range(1, self.x_length - 1):
            dx2 = np.square(diff[i])
            if dx2 <= 1e-8:
                ticks.append(self.x_axis[i])
            # if abs(dx) < absMin:
            #     absMin = abs(dx)

        show_plot(self.Entropy, 
                  "Harmonic Entropy sqrt(nd) weighting", 
                  "Dyad (cents)", 
                  "Dissonance", 
                #   annotate="s={}, N<{}, a={}, {}c<{}".format(self.s, self.N, self.a, self.res, self.limit),
                  annotate="s={}, N<{}, a={}".format(self.s, self.N, self.a),
                #   xticks=ticks,
                #   plotDiffs=diff,
                #   plotDiffs2=diff2
                  )


if __name__ == "__main__":

    eightoctaves=ratio_to_cents(2**2)

    he = HarmonicEntropy(s=17, N=1000, a=3)
    # he.calculate()
    he.prepareBasis()
    # he.setOddBasis()
    he.convolveHRE()
    he.plot()
