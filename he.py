"""

  https://en.xen.wiki/w/Harmonic_entropy
  help from https://gist.github.com/Sin-tel/a0279a2fe758e5a79496ba182d4ed992
            https://gist.github.com/Sin-tel/8d1a55a0e34ca159ac6aa61e325648d2

"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

from utils import *
from basis import *

def show_plot(data, title="Plot", xlabel="X", ylabel="Y", **kwArgs):
    default_label = ylabel
    if "label-default" in kwArgs:
        default_label = kwArgs["label-default"]
        del kwArgs["label-default"]
    if "imshow" in kwArgs:
        cmap = None if "cmap" not in kwArgs else kwArgs["cmap"]
        if "figsize" in kwArgs:
            plt.figure(figsize=kwArgs["figsize"])
        plt.imshow(data, cmap=cmap, aspect="equal", origin="lower")
        plt.axis('off')
    else:
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

    if "filename" in kwArgs:
        print(1, "Writing: " + kwArgs["filename"])
        plt.savefig(kwArgs["filename"], dpi=kwArgs["dpi"], pad_inches=0, bbox_inches='tight')

    plt.show()
    return plt

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
    weight_option = None

    kwArgs = None
    verbose = 1

    # state
    x_axis = None

    period_harmonic = None
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

    updateBasis = None
    recalculate = None

    Entropy = None
    EntropyAltWeights = None

    plotted = False

    # pre-computes
    i_ss2 = None
    ssqrt2pi = None

    tri_y_scalar = np.sqrt(3) / 2

    def __init__(self, spread=17, N=1000, res=1, limit=2400, alpha=8, weight=None, he3=False, **kwArgs):
        if weight is None:
            weight = 'default'
        self.he3 = he3
        self.kwArgs = kwArgs
        if "verbose" in self.kwArgs:
            self.verbose = int(self.kwArgs["verbose"])

        self.update(spread, N, res, limit, alpha)

        if self.he3:
            self.prepare3heBasis()
        else:
            self.prepareBasis()

        self.setWeightingOption(weight)

    def vprint(self, level, str):
        if self.verbose >= level:
            print(str)

    def suffix(self):
        tokens = {}
        if self.he3:
            tokens["he3"] = ""
        
        tokens["s"] = self.s
        tokens["a"] = self.a
        tokens["N"] = self.N

        if self.res != 1:
            tokens["c"] = self.res
        if self.limit != 1200:
            tokens["max"] = self.limit
        if self.weight_func_name is not None:
            tokens["wt"] = f'{self.weight_func_name}-'

        return "_".join([ f'{tokens[k]}{k}' for k in tokens ])

    def update(self, s=None, N=None, res=None, limit=None, a=None, weight=None):
        updated = False
        if N is not None:
            updated = updated or self.N != N
            self.N = N
            self.updateBasis = True
        if res is not None:
            updated = updated or self.res != res
            self.res = res
            self.updateBasis = True
        if limit is not None:
            updated = updated or self.limit != limit
            self.limit = limit
            self.updateBasis = True
        if weight is not None:
            updated = updated or self.weight_option != weight
            self.setWeightingOption(weight)
            self.recalculate = True
        if a is not None:
            updated = updated or self.a != a
            self.a = a
            self.recalculate = True
        if s is not None:
            updated = updated or self.s != s
            self.s = s
            self.recalculate = True

        if updated:
            if res is not None or limit is not None:
                self.x_axis = np.arange(0, int(np.ceil(self.limit+self.res)), step=self.res)
                self.x_length = len(self.x_axis)

            if limit is not None:
                self.period_harmonic = int(np.round(np.exp2(self.limit / 1200)))

            if s is not None:
                self.i_ss2 = 1 / (self.s**2 * 2)
                self.ssqrt2pi = self.s * 2.50662827463

            if self.updateBasis:
                self.recalculate = True

        return updated

    def prepareBasis(self):
        file = os.path.join(os.path.dirname(__file__), "he_data", "farey{}.npy".format(self.N))
        if os.path.exists(file):
            basis_set = np.load(file)
        else:
            self.vprint(1, "Calculating rationals...")
            basis_set = farey(self.N)
            np.save(file, basis_set)

        self.basis_set      = basis_set
        self.basis_periods  = farey_set_to_basis(self.basis_set, self.period_harmonic)
        self.basis_length   = self.basis_periods.shape[0]

        # self.basis_ratios   = np.round(self.basis_periods[:, 0] / self.basis_periods[:, 1], RATIO_ROUND)
        self.basis_cents    = nd_basis_to_cents(self.basis_periods).ravel()

    def prepare3heBasis(self):

        default_c_limit = 27_000_000
        c_limit = default_c_limit

        file_basename = "he3_basis"
        file_params = { "N": self.N, "h": self.period_harmonic}
        if (c_limit != default_c_limit):
            file_params["climit"] = c_limit
        
        file_suffix = "_".join([ f'{file_params[kw]}{kw}' for kw in file_params ]) + ".npy"
        
        file = os.path.join(os.path.dirname(__file__), "he_data", file_basename + file_suffix)
        if os.path.exists(file):
            triplets = np.load(file)
        else:
            self.vprint(1, "Calculating rationals...")
            triplets = get_triplet_basis(self.N, self.period_harmonic)
            np.save(file, triplets)

        self.vprint(1, f"Preparing basis...")
        self.basis_triad_set = np.asarray(triplets)
        self.basis_length = self.basis_triad_set.shape[0]
        self.vprint(1, f"\tSet size: {self.basis_length}")

        # self.basis_triad_ratios = np.ones((self.basis_triad_set.shape[0], 2))
        # self.basis_triad_ratios[:,0] = self.basis_triad_set[:,1] / self.basis_triad_set[:,0]
        # self.basis_triad_ratios[:,1] = self.basis_triad_set[:,2] / self.basis_triad_set[:,1]

        self.basis_triad_cents = nd_basis_to_cents(self.basis_triad_set)

        self.basis_triad_x = np.round((self.basis_triad_cents[:,0] + (self.basis_triad_cents[:,1] / 2)) / self.res).astype(np.int64)
        self.basis_triad_y = np.round(self.basis_triad_cents[:,1] * self.tri_y_scalar / self.res).astype(np.int64)
        self.basis_triad_xy = (self.basis_triad_y, self.basis_triad_x)

    def setBasisMask(self, mask):
        basis_n = np.extract(mask, self.basis_periods[:,0])
        basis_d = np.extract(mask, self.basis_periods[:,1])
        self.basis_periods  = np.column_stack((basis_n, basis_d))
        # self.basis_ratios   = np.extract(mask, self.basis_ratios)
        self.basis_cents    = np.extract(mask, self.basis_cents)
        self.basis_length   = len(self.basis_cents)

        if self.he3:
            # self.basis_periods[:,2] = np.extract(mask, self.basis_periods[:,2])
            # self.basis_triad_pedals = np.extract(mask, self.basis_triad_pedals)
            # self.basis_triad_ratios = np.extract(mask, self.basis_triad_ratios)
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
        self.option = option
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
            self.option = None
            raise Exception("Unknown weighing option: " + str(option))

    def setDefaultWeightingFunction(self):
        self.setWeightingFunction(None)
    
    def setLenCfWeight(self):
        weigh_ratio = lambda ratio: len(get_cf(ratio))
        weigh = lambda a: np.prod(np.vectorize(weigh_ratio)(a[:, 1:] / a[:,:-1]), axis=1)
        self.setWeightingFunction(weigh, "lencf")

    def setLenMaxCfWeight(self):
        weigh_cf = lambda cf: len(cf) * max(cf)
        weigh_ratio = lambda ratio: weigh_cf(get_cf(ratio))
        weigh = lambda a: np.sqrt(np.prod(np.vectorize(weigh_ratio)(a[:, 1:] / a[:,:-1]), axis=1))
        self.setWeightingFunction(weigh, "sqrt(len(cf)*max(cf))")

    def setSumCfWeight(self):
        weigh_ratio = lambda ratio: sum(get_cf(ratio))
        weigh = lambda a: np.sqrt(np.prod(np.vectorize(weigh_ratio)(a[:, 1:] / a[:,:-1]), axis=1))
        self.setWeightingFunction(weigh, "sum(cf)")

    def convolveHRE(self):
        if self.weight_func is None:
            base_weights = 1 / np.sqrt(np.prod(self.basis_periods, axis=1))
        else:
            base_weights = 1 / self.weight_func(self.basis_periods)

        # turn basis cents into a sum of delta functions
        basis_index = np.rint(self.basis_cents / self.res).astype(np.uint32)

        K = np.zeros(self.x_length)
        np.add.at(K, basis_index, base_weights)
        Ka = np.zeros(self.x_length)
        np.add.at(Ka, basis_index, np.power(base_weights, self.a))

        s_range = np.round(self.s * 5)
        xs = np.arange(-s_range, s_range, 1)
        S = np.exp(-(xs**2) * self.i_ss2)

        psi = signal.convolve(K, S, 'same')
        pa = signal.convolve(Ka, S ** self.a, 'same')

        sigma = 1e-16
        alpha = self.a
        if alpha == 1:
            alpha = sigma

        return np.log(pa / (psi ** alpha + sigma) + sigma) / (1 - alpha)
    
    def convolve3HRE(self):
        self.vprint(1, "Weighing...")
        if self.weight_func is None:
            base_weights = 1 / np.sqrt(np.prod(self.basis_triad_set, axis=1))
        else:
            base_weights = 1 / self.weight_func(self.basis_triad_set)

        K = np.zeros(shape=(self.x_length, self.x_length))
        np.add.at(K, self.basis_triad_xy, base_weights)
        
        Ka = np.zeros(shape=(self.x_length, self.x_length))
        np.add.at(Ka, self.basis_triad_xy, base_weights ** self.a)

        self.vprint(1, "Smoothing...")
        s_range = round(self.s*5)
        sx = np.arange(-s_range, s_range, self.res)
        sy = np.arange(-s_range, s_range, self.res)
        x, y = np.meshgrid(sx, sy)
        S = np.exp(-((x**2 + y**2) * self.i_ss2))

        self.vprint(1, "Convolving...")
        psi = signal.convolve(K, S, mode = 'same')
        pa = signal.convolve(Ka, S**self.a, mode = 'same')

        sigma = 1e-16
        alpha = self.a
        if alpha == 1:
            alpha = sigma

        entropy = np.log((pa + sigma) / ((psi ** alpha) + sigma)) / (1 - alpha)

        self.vprint(1, "Masking...")
        # clean up
        gx = np.arange(0, self.x_length, 1)
        gy = np.arange(0, self.x_length, 1)
        mx, my = np.meshgrid(gx, gy)

        mask = mx - my / np.sqrt(3) > 0
        mask &= self.x_length - mx - my / np.sqrt(3) > 0

        entropy_masked = 7 - entropy
        entropy_masked[~mask] = 0
        return entropy_masked
        
    def calculate(self, loadFile=True):
        if self.he3:
            file = os.path.join(os.path.dirname(__file__), "he_data", "3he_{}.npy".format(self.suffix()))
            if loadFile and os.path.exists(file):
                self.Entropy = np.load(file)
            else:
                self.Entropy = self.convolve3HRE()

            self.recalculate = False
            return

        file = os.path.join(os.path.dirname(__file__), "he_data", "he_{}.npy".format(self.suffix()))
        if loadFile and os.path.exists(file):
            self.Entropy = np.load(file)
        else:
            self.Entropy = self.convolveHRE()
        
        self.recalculate = False

    def writeEntropy(self):
        if self.he3:
            file = os.path.join(os.path.dirname(__file__), "he_data", "3he_{}".format(self.suffix()))
        else:
            file = os.path.join(os.path.dirname(__file__), "he_data", "he_{}".format(self.suffix()))
        self.vprint(1, f"Writing: {file}")
        np.save(file, self.Entropy)
        np.savetxt(file+".txt", self.Entropy,  fmt="%f")

    def getEntropyPlotData(self, min_cents=None, max_cents=None):
        if self.he3:
            pass # todo
            return (self.x_axis, self.Entropy)

        startX = 0
        endX = self.x_length

        if min_cents is not None:
            startX = max(0, int(np.round(min_cents / self.res)))

        if max_cents is not None:
            endX = min(self.x_length, int(np.round(max_cents / self.res)))

        return (self.x_axis[startX:endX], self.Entropy[startX:endX])

    def plot(self, save=True):
        self.vprint(1, "Plotting...")

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

        if save:
            filename = os.path.join("he_plots", f'3he_{self.suffix()}')
            plotArgs["filename"] = getUniqueFilename(filename, "png")
            plotArgs["dpi"] = 480

        if self.he3:
            plotArgs["imshow"]=True
            plotArgs["figsize"]=(16,16)
            plotArgs["cmap"]="inferno"

        self.plotted = True

        return show_plot(plot_data, 
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

    if plot or not save:
        he.plot()
