"""

  https://en.xen.wiki/w/Harmonic_entropy

"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

from utils import *

CENTS_ROUND=3
RATIO_ROUND=6

def farey_set_to_basis(ndSet, octaves=1):
    size = ndSet.shape[0]
    period_shape = (size * octaves - int(octaves > 1), 2)

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
    default_label = ylabel
    if "label-default" in kwArgs:
        default_label = kwArgs["label-default"]
        del kwArgs["label-default"]

    plt.plot(data, label=default_label)
    num_plots=1

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if kwArgs:
        for kw in kwArgs:
            if kw == "annotate":
                txtpos = (0, max(data))
                plt.annotate(kwArgs[kw], txtpos)
            elif kw == "xticks":
                labels=None
                if "xtickLabels" in kwArgs:
                    labels = kwArgs["xtickLabels"]
                plt.xticks(kwArgs[kw], labels=labels)
            elif kw.startswith("plot"):
                if kw.endswith("label"):
                    continue
                
                labelKw = f'{kw}-label'
                label = kwArgs[labelKw]
                plt.plot(kwArgs[kw], label=label)
                num_plots += 1

    if num_plots > 1:
        plt.legend()
        # primary_plot.set_label(kwArgs['label-default'])

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

    he3 = False

    basis_filter = None

    weight_func = None
    weight_func_name = None

    kwArgs = None

    # state
    x_axis = None

    num_periods = None
    basis_set = None
    basis_periods = None
    basis_ratios = None
    basis_cents = None

    basis_triad_pedals = None
    basis_triad_cents = None
    basis_triad_x = None
    basis_triad_y = None
    basis_triad_xy = None

    x_length = None
    basis_length = None

    Entropy = None
    EntropyAltWeights = None

    # pre-computes
    i_ss2 = None
    ssqrt2pi = None

    tri_y_scalar = np.sqrt(3) / 2

    def __init__(self, spread=17, N=1000, res=1, limit=2400, alpha=8, weight=None, he3=False, **kwArgs):
        if weight is None:
            weight = 'default'
        self.he3 = he3
        self.kwArgs = kwArgs
        self.update(spread, N, res, limit, alpha)
        self.prepareBasis()
        self.setWeightingOption(weight)

    def suffix(self):
        s = "{}s_{}a_{}N_{}c_{}max".format(self.s, self.a, self.N, self.res, self.limit)
        if self.weight_func_name is not None:
            s += "_" + self.weight_func_name
        return s

    def update(self, s=None, N=None, res=None, limit=None, a=None, weight=None):
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
        if weight is not None:
            self.setWeightingOption(weight)

        self.x_axis = np.arange(0, int(np.ceil(limit+res)), step=res)
        self.x_length = len(self.x_axis)

        self.num_periods = int(np.ceil(self.limit / 1200))

        self.i_ss2 = 1 / (self.s**2 * 2)
        self.ssqrt2pi = self.s * 2.50662827463

    def prepareBasis(self, basis=None):
        if basis:
            self.basis_set = basis["set"]
            self.basis_periods = basis["periods"]
            self.basis_ratios = basis["ratios"]
            self.basis_cents = basis["cents"]
            self.basis_length = len(self.basis_cents)
            return
        
        file = os.path.join(os.path.dirname(__file__), "he_data", "farey{}.npy".format(self.N))
        if os.path.exists(file):
            basis = np.load(file)
        else:
            print("Calculating rationals...")
            basis = farey(self.N)
            np.save(file, basis)

        self.updateBasisSet(basis)

    def updateBasisSet(self, newBasis):
        self.basis_set      = newBasis
        self.basis_periods  = farey_set_to_basis(self.basis_set, self.num_periods)

        self.basis_ratios   = np.round(self.basis_periods[:, 0] / self.basis_periods[:, 1], RATIO_ROUND)
        self.basis_cents    = np.vectorize(ratio_to_cents)(self.basis_ratios)
        self.basis_length   = len(self.basis_cents)

        if self.he3:
            self.basis_triad_pedals = np.exp2(np.trunc(np.log2(self.basis_periods[:, 1])))
            self.basis_periods = np.append(self.basis_periods, self.basis_triad_pedals.T[:, None], 1)
            self.basis_triad_ratios = np.round(self.basis_periods[:, 1] / self.basis_periods[:, 2], RATIO_ROUND)
            self.basis_triad_cents = np.vectorize(ratio_to_cents)(self.basis_triad_ratios)
            mask = self.basis_triad_cents <= self.limit
            filtered_cents = np.extract(mask, self.basis_triad_cents)
            self.basis_triad_x = np.round(filtered_cents + (filtered_cents / 2)).astype(np.int64)
            self.basis_triad_y = np.round(filtered_cents * self.tri_y_scalar).astype(np.int64)
            self.basis_triad_xy = (np.extract(self.basis_triad_x <= self.limit, self.basis_triad_x), np.extract(self.basis_triad_y <= self.limit, self.basis_triad_y))

    def setBasisMask(self, mask):
        basis_n = np.extract(mask, self.basis_periods[:,0])
        basis_d = np.extract(mask, self.basis_periods[:,1])
        self.basis_periods  = np.column_stack((basis_n, basis_d))
        self.basis_ratios   = np.extract(mask, self.basis_ratios)
        self.basis_cents    = np.extract(mask, self.basis_cents)
        self.basis_length   = len(self.basis_cents)

        if self.he3:
            self.basis_periods[:,2] = np.extract(mask, self.basis_periods[:,2])
            self.basis_triad_pedals = np.extract(mask, self.basis_triad_pedals)
            self.basis_triad_ratios = np.extract(mask, self.basis_triad_ratios)
            self.basis_triad_cents = np.extract(mask, self.basis_triad_cents)
            self.basis_triad_x = np.extract(mask, self.basis_triad_x)
            self.basis_triad_y = np.extract(mask, self.basis_triad_y)
            self.basis_triad_xy = (self.basis_triad_x, self.basis_triad_y)

    def setOddBasis(self):
        mask = self.basis_periods[:, 0] & self.basis_periods[:, 1] & 1
        self.setBasisMask(mask)

    def setWeightingFunction(self, func, name=None):
        self.weight_func = func
        if func is None:
            self.weight_func_name = None
        else:
            self.weight_func_name = "wx" if name is None else name

    def setWeightingOption(self, option):
        if option == 'default' or option is None or option == 'sqrtnd':
            self.setDefaultWeightingFunction()
        elif option == 'lencf':
            self.setLenCfWeight()
        elif option == 'lenmaxcf':
            self.setLenMaxCfWeight()
        elif option == 'sumcf':
            self.setSumCfWeight()
        elif option == 'all':
            self.weight_func_name = option
            self.EntropyAltWeights = {}

            self.setDefaultWeightingFunction()
            self.Entropy = self.convolveHRE()
            self.setLenCfWeight()
            self.EntropyAltWeights['lencf'] = self.convolveHRE()
            self.setLenMaxCfWeight()
            self.EntropyAltWeights['lenmaxcf'] = self.convolveHRE()
            self.setSumCfWeight()
            self.EntropyAltWeights['sumcf'] = self.convolveHRE()

            self.setDefaultWeightingFunction()

        else:
            raise Exception("Unknown weighing option: " + str(option))

    def setDefaultWeightingFunction(self):
        self.setWeightingFunction(None)
    
    def setLenCfWeight(self):
        weigh_nd = lambda ns,ds: [ len(get_cf(ns[i]/ds[i])) for i in range(len(ns)) ]
        def weigh(array):
            shape = (self.basis_periods.shape[0], self.basis_periods.shape[1]-1)
            weights = np.zeros(shape)
            for c in range(shape[1]):
                weights[:,c] = weigh_nd(array[:,c], array[:,c+1])
            return weights
        self.setWeightingFunction(weigh, "lencf")

    def setLenMaxCfWeight(self):
        prod = lambda cf: len(cf) * max(cf)
        prod_nd = lambda ns,ds: [ prod(get_cf(ns[i]/ds[i])) for i in range(len(ns)) ]
        def weigh(array):
            shape = (self.basis_periods.shape[0], self.basis_periods.shape[1]-1)
            weights = np.zeros(shape)
            for c in range(shape[1]):
                weights[:,c] = np.sqrt(prod_nd(array[:,c], array[:,c+1]))
            return weights
        self.setWeightingFunction(weigh, "sqrt(len(cf)*max(cf))")

    def setSumCfWeight(self):
        sum_nd = lambda ns,ds: [ sum(get_cf(ns[i]/ds[i])) for i in range(len(ns)) ]
        def weigh(array):
            shape = (self.basis_periods.shape[0], self.basis_periods.shape[1]-1)
            weights = np.zeros(shape)
            for c in range(shape[1]):
                weights[:,c] = np.asarray(sum_nd(array[:,c], array[:,c+1]))
            return weights
        self.setWeightingFunction(weigh, "sum(cf)")

    def convolveHRE(self):
        if self.weight_func is None:
            base_weights = 1 / np.sqrt(np.prod(self.basis_periods, axis=1))
        else:
            base_weights = 1 / self.weight_func(self.basis_periods)

        base_weighted_alpha = np.power(base_weights, self.a)

        if self.he3:
            K = np.zeros(shape=(self.x_length, self.x_length))
            Ka = np.zeros(shape=(self.x_length, self.x_length))
            
            # splat dirac deltas into buffer
            np.add.at(K, self.basis_triad_xy, base_weights)
            np.add.at(Ka, self.basis_triad_xy, base_weights**alpha)
            plt.figure(figsize=(6,6))
            plt.imshow(np.log(1e-6+K), aspect="equal", origin='lower')
            plt.axis('off')
            plt.show()
            return

        # turn basis cents into a sum of delta functions
        basis_index = np.rint(self.basis_cents / self.res).astype(np.uint32)

        K = np.zeros(self.x_length)
        Ka =  np.zeros(self.x_length)
        for ji in range(self.basis_length):
            di = basis_index[ji]
            if di < 0 or di >= self.x_length:
                print(f"Warning: ignoring basis overflow with {self.basis_periods[ji,0]}/{self.basis_periods[ji,1]}")
                continue
            K[di] += base_weights[ji]
            Ka[di] += base_weighted_alpha[ji]

        # S = np.exp(-np.square(self.x_axis - self.x_length//2) * self.i_ss2) / self.ssqrt2pi
        # s_range = round(self.s * 5)
        # xr=np.arange(-self.ssqrt2pi, self.ssqrt2pi, self.res)
        S = np.exp(-np.arange(-self.ssqrt2pi, self.ssqrt2pi, self.res)**2 * self.i_ss2)
        # plt.plot(xr,S)
        # plt.show()
        # return
        sigma = 1e-16
        alpha = self.a
        if alpha == 1:
            alpha = sigma

        psi = signal.convolve(K, S, 'same')
        pa = signal.convolve(Ka, S ** alpha, 'same')

        return np.log(pa / (psi ** alpha) + sigma) / (1 - alpha)

    def calculate(self, loadFile=True):
        file = os.path.join(os.path.dirname(__file__), "he_data", "he_{}.npy".format(self.suffix()))
        if loadFile and os.path.exists(file):
            self.Entropy = np.load(file)
            return

        self.Entropy = self.convolveHRE()

    def writeEntropy(self):
        file = os.path.join(os.path.dirname(__file__), "he_data", "he_{}".format(self.suffix()))
        print(f"Writing: {file}")
        np.save(file, self.Entropy)
        np.savetxt(file+".txt", self.Entropy,  fmt="%f")

    def plot(self):
        plot_data = self.Entropy

        plotArgs = {}

        ticksKw = 'ticks'
        if ticksKw in self.kwArgs and self.kwArgs[ticksKw]:
            minima_index = signal.argrelextrema(self.Entropy, np.less)[0]
            minima_entropy = self.Entropy[minima_index]

            bins = int(5 / self.res)
            bins += (1 - bins % 2)
            hist, edges = np.histogram(minima_entropy, bins)

            tick_edge = int(np.ceil(bins/2))
            max_entropy = edges[tick_edge]
            minima_ticks = []
            for m in minima_index:
                if self.Entropy[m] > max_entropy:
                    continue
                minima_ticks.append(m)
            
            ticks = [0, *minima_ticks, self.x_length - 1]
            tickLabels = [ f"{(t*self.res):.2f}" for t in ticks ]
            plotArgs["xticks"] = ticks
            plotArgs["xtickLabels"] = tickLabels

        weight_name = "sqrt(nd)" 
        if self.weight_func_name is not None:
            weight_name = self.weight_func_name

        plotArgs['label-default'] = weight_name

        if self.EntropyAltWeights is not None:
            for kw in self.EntropyAltWeights:
                key = f'plot-{kw}'
                plotArgs[key] = self.EntropyAltWeights[kw]
                plotArgs[f'{key}-label'] = kw
            
            weight_name = "all"

        title = "Harmonic Entropy {} weighting".format(weight_name)

        annotation = f"s={self.s}, N<{self.N}, a={self.a}"
        if self.res != 1:
            annotation += f", res={self.res}c"

        plotArgs["annotate"] = annotation

        show_plot(plot_data, 
                  title, 
                  "Dyad (cents)", 
                  "Dissonance",
                  **plotArgs
                  )


if __name__ == "__main__": 
    parser = argparse.ArgumentParser("harmonicentro.py")
    parser.add_argument('-n', '--N', type=int, help="Exclusive limit of basis set ratio denominators", default=1000)
    parser.add_argument('-s', '--spread',type=float, help="Bandwidth of spreading function in cents")
    parser.add_argument('-a', '--alpha', type=float, help="Order of weight scaling", default=3)
    parser.add_argument('-r', '--res', type=float, help="Resolution of x-axis in cents")
    parser.add_argument('-l', '--limit', type=float, help="Last cents value to calculate")
    parser.add_argument('-w', '--weight', choices=['default', 'sqrtnd', 'lencf', 'lenmaxcf', 'sumcf', 'all'])
    parser.add_argument('--he3', action='store_true', help='3HE mode')
    parser.add_argument('--plot', action='store_true', help="Display plot")
    parser.add_argument('--ticks', action='store_true', help="Auto-select minima-based x-axis ticks")
    parser.add_argument('--save', action='store_true', help="Save to file")

    parsed = parser.parse_args()

    options = vars(parsed)
    save = options['save']
    del options['save']
    plot = options['plot']
    del options['plot']

    heArgs = { k:v for k,v in options.items() if v is not None }
    he = HarmonicEntropy(**heArgs)
    he.calculate()

    if save:
        he.writeEntropy()

    if plot:
        he.plot()
