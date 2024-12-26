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

def create_plot(data, title="Plot", xlabel="X", ylabel="Y", **kwArgs):
    default_label = ylabel
    if "label-default" in kwArgs:
        default_label = kwArgs["label-default"]
        del kwArgs["label-default"]
    if "imshow" in kwArgs:
        cmap = None if "cmap" not in kwArgs else kwArgs["cmap"]
        if "figsize" in kwArgs:
            plt.figure(figsize=kwArgs["figsize"])
        origin = "lower" if "origin" not in kwArgs else kwArgs["origin"]
        plt.imshow(data, cmap=cmap, aspect="equal", origin=origin)
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

    return plt



class HarmonicEntropy:
    s = None
    N = None
    res = None
    limit = None
    a = None

    he3 = False

    basis_mask = None
    basis_mask_func = None
    basis_mask_name = None
    basis_mask_option = None

    basis_transform_func = None
    basis_transform_option = None
    entropy_mask_fnc = None
    entropy_mask = None

    weight_func = None
    weight_func_name = None
    weight_option = None

    kwArgs = None
    verbose = 1
    output_dir=None

    # state
    x_axis = None
    x_length = None

    period_harmonic = None
    basis_set = None
    basis_periods = None
    basis_length = None

    basis_ratios = None
    basis_cents = None
    
    basis_weights = None
    basis_weight_alphas = None
    basis_transform = None
    basis_distribution = None
    basis_distribution_alpha = None
    basis_spread = None
    basis_spread_alphas = None

    updateX = None
    regenBasis = None
    updateBasis = None
    updateWeights = None
    updateTransform = None
    updateDistribution = None
    updateSpread = None
    updateAlpha = None
    updateMask = None

    loadedEntropyFile = None

    Entropy = None
    EntropyAltWeights = None

    plot = None

    # pre-computes
    i_ss2 = None
    ssqrt2pi = None
    tri_y_scalar = np.sqrt(3) / 2

    # constants
    BASIS_MASK_OPTIONS = ["pl", "pr", "odd", "primes", "composites"]


    def __init__(self, spread=15, N=100, res=1, limit=2400, alpha=7, he3=False, **kwArgs):
        
        # Global Params
        self.kwArgs = kwArgs

        if "verbose" in self.kwArgs:
            self.verbose = int(self.kwArgs["verbose"])
        
        if "out" in self.kwArgs:
            self.output_dir = kwArgs["out"]

        # Model Parameters
        args = { 
            "N": N,
            "limit": limit,
            "res": res,
            "s": spread,
            "a": alpha,
            }

        self.he3 = he3 

        if "weight" in self.kwArgs:
            args["weight"] = self.kwArgs["weight"]

        if "tx" in self.kwArgs:
            args["tx"] = self.kwArgs["tx"]
        else:
            self.setTransformOption("default")

        for option in self.BASIS_MASK_OPTIONS:
            if option in self.kwArgs:
                args[option] = True
                break

        self._vprint(2, f"args: {args}")
        self.update(args)


    def _vprint(self, level, str):
        if self.verbose >= level:
            print(str)

    def suffix(self):
        tokens = {}
        if self.he3:
            tokens["he3"] = ""
        
        tokens["s"] = str(self.s)
        tokens["a"] = str(self.a)
        tokens["N"] = str(self.N)

        if self.basis_mask_name is not None:
            tokens["N"] += '-' + self.basis_mask_name

        if self.res != 1:
            tokens["c"] = self.res
        if self.limit != 1200:
            tokens["max"] = self.limit
        if self.weight_func_name is not None:
            tokens["wt"] = f'{self.weight_func_name}-'
        if self.basis_transform_option is not None:
            tokens["tx"] = f'{self.basis_transform_option}-'

        return "_".join([ f'{tokens[k]}{k}' for k in tokens ])

    def update(self, args):
        updated = False
        for kw in args: # if needed, map callbacks to dict to eliminate conditions
            value = args[kw]
            if kw == "N":
                updated = updated or self.N != value
                self.N = args["N"]
                self.regenBasis = True
            elif kw == "res":
                updated = updated or self.res != value
                self.res = args["res"]
                self.updateX = True
            elif kw == "limit":
                updated = updated or self.limit != value
                self.limit = args["limit"]
                self.updateX = True
                self.regenBasis = True
            elif kw == "weight":
                updated = updated or self.weight_option != value
                self.setWeightingOption(value)
            elif kw == "tx":
                updated = updated or self.basis_transform_option != value
                self.setTransformOption(value)
            elif kw == "a":
                updated = updated or self.a != value
                self.a = value
                self.updateAlpha = True
            elif kw == "s":
                updated = updated or self.s != value
                self.s = value
                self.updateSpread = True
            elif kw in self.BASIS_MASK_OPTIONS:
                if self.basis_mask_option != kw:
                    updated = True
                    self.setBasisMaskOption(kw)

        return updated
    
    def _prepare_x_axis(self):
        self.x_axis = np.arange(0, int(np.ceil(self.limit+self.res)), step=self.res)
        self.x_length = len(self.x_axis)

        self.period_harmonic = int(np.round(np.exp2(self.limit / 1200)))

        self.updateX = False
        self.updateBasis = True
        self.updateTransform = True
        self.updateMask = True

    def _generate_basis(self):
        if not self.he3:
            file = os.path.join(os.path.dirname(__file__), "he_data", "farey{}.npy".format(self.N))
            if os.path.exists(file):
                basis_set = np.load(file)
            else:
                self._vprint(1, "Calculating rationals...")
                basis_set = farey(self.N)
                np.save(file, basis_set)

            self.basis_set      = basis_set
        else:
            default_c_limit = 27_000_000
            c_limit = default_c_limit

            file_basename = "he3_basis"
            file_params = { "N": self.N, "h": self.period_harmonic}
            if (c_limit != default_c_limit):
                file_params["climit"] = c_limit
            
            file_suffix = "_".join([ f'{file_params[kw]}{kw}' for kw in file_params ]) + ".npy"
            
            file = os.path.join(os.path.dirname(__file__), "he_data", file_basename + file_suffix)
            if os.path.exists(file):
                self.basis_set = np.load(file)
            else:
                self._vprint(1, "Calculating rationals...")
                self.basis_set = get_triplet_basis(self.N, self.period_harmonic)
                np.save(file, self.basis_set)
        
        self.regenBasis = False
        self.updateBasis = True

    def _prepare_basis_periods(self):
        self._vprint(1, f"Preparing basis...")

        if self.he3:
            self.basis_periods = np.asarray(self.basis_set).copy()
        else:
            self.basis_periods  = farey_set_to_basis(self.basis_set, self.period_harmonic)

        self.basis_length = self.basis_periods.shape[0]
        self.basis_cents = nd_basis_to_cents(self.basis_periods)

        self.updateBasis = False
        self.updateWeights = True
        self.updateTransform = True
        if self.basis_distribution is None:
            newShape = self.x_length
            if self.he3:
                newShape = (self.x_length, self.x_length)
            self.basis_distribution = np.zeros(newShape)
            self.basis_distribution_alpha = np.zeros(newShape)

    def setBasisMask(self, mask, reprepare=True):
        if mask is None or reprepare:
            self._prepare_basis_periods()

        if mask is not None:
            self.basis_periods  = np.compress(mask, self.basis_periods, 0)
            self.basis_length   = self.basis_periods.shape[0]
            self.basis_cents    = np.compress(mask, self.basis_cents, 0)

        self.basis_mask = mask

        self.updateWeights = True
        self.updateTransform = True

    def createBasisMask(self, basisTest, reprepare=True):
        if self.updateX:
            self._prepare_x_axis()
        if self.regenBasis:
            self._generate_basis()
        if self.updateBasis or reprepare:
            self._prepare_basis_periods()
        self._vprint(1, "Masking basis set...")
        return create_mask_from_lambdas(self.basis_periods, basisTest)

    def setBasisMaskOption(self, option, reprepare=True):
        mask_func = None
        name = None

        if option == "default" or option == "all" or option is None:
            pass
        elif option == "odd":
            name = "Odds"
            mask_func = create_exclusive_prime_test([2])
        elif option == "composites":
            name = "Composites"
            primeTest = create_is_prime_test()
            mask_func = lambda basis: ~primeTest(basis)
        elif option == "primes":
            name = "StrictPrimes"
            mask_func = create_is_prime_test()
        else:
            raise Exception("Unknown basis mask option")
            
        self.basis_mask_name = name
        self.basis_mask_func = mask_func
        self.basis_mask = self.createBasisMask(self.basis_mask_func, reprepare)
        self.setBasisMask(self.basis_mask, False)

    def setWeightingOption(self, option, **kwArgs):
        if option != 'custom' and option == self.weight_option:
            return
        weigh = None
        name = None
        self.updateWeights = True

        if option == 'default' or option is None or option == 'sqrtnd':
            name = 'sqrt(nd)'
        elif option == 'lencf':
            weigh_ratio = lambda ratio: len(get_cf(ratio))
            weigh = lambda a: np.prod(np.vectorize(weigh_ratio)(a[:, 1:] / a[:,:-1]), axis=1)
            name = "lencf"
        elif option == 'lenmaxcf':
            weigh_cf = lambda cf: len(cf) * max(cf)
            weigh_ratio = lambda ratio: weigh_cf(get_cf(ratio))
            weigh = lambda a: np.sqrt(np.prod(np.vectorize(weigh_ratio)(a[:, 1:] / a[:,:-1]), axis=1))
            name = "sqrt(len(cf)*max(cf))"
        elif option == 'sumcf':
            weigh_ratio = lambda ratio: sum(get_cf(ratio))
            weigh = lambda a: np.sqrt(np.prod(np.vectorize(weigh_ratio)(a[:, 1:] / a[:,:-1]), axis=1))
            name = "sum(cf)"
        elif option == 'all':
            name = 'all'
        elif option == 'custom':
            weigh = kwArgs['weigh'] # function
            if "name" in kwArgs:
                name = kwArgs['name']
        else:
            self.weight_func = None
            self.weight_option = None
            self.weight_func_name = None
            raise Exception("Unknown weighting option: " + str(option))

        self.weight_option = option
        self.weight_func = weigh
        self.weight_func_name = name

    def _weight_basis(self):
        self._vprint(1, 'Weighting...')

        if self.updateWeights:
            if self.weight_func is None:
                self.basis_weights = np.reciprocal(np.sqrt(np.prod(self.basis_periods, axis=1)))
            else:
                self.basis_weights = np.reciprocal(self.weight_func(self.basis_periods))

        if self.updateWeights or self.updateAlpha:
            self.basis_weight_alphas = self.basis_weights ** self.a

        self.updateWeights = False
        self.updateDistribution = True

    def _get_mask_mesh(self):
        return np.meshgrid(np.arange(0, self.x_length, 1), np.arange(0, self.x_length, 1))
        
    def setTransformOption(self, option, **kwArgs):
        if option != 'custom' and option == self.basis_transform_option:
            return
        tx = None
        name = None
        mask_fnc = None
        self.updateTransform = True

        if option == 'default':
            if self.he3:
                def create_mask():
                    mx, my = self._get_mask_mesh()
                    my_sqrt3 = my / np.sqrt(3)
                    return (mx - my_sqrt3 > 0) & (self.x_length - mx - my_sqrt3 > 0)
                mask_fnc = create_mask
                
        elif option == 'linear' or option == 'lin':
            if self.he3:
                def linHe3Tx(periods):
                    indexes = np.rint((periods[:,1:] / periods[:,:-1] - 1) / self.period_harmonic * (self.x_length - 1))
                    tx = np.rint((indexes[:,0] + (indexes[:,1] / 2)) / self.res).astype(np.uint32)
                    ty = np.rint(indexes[:,1] * self.tri_y_scalar / self.res).astype(np.uint32)
                    return (tx, ty)
                tx = linHe3Tx
            else:
                tx = lambda periods: np.rint(((periods[:,1] / periods[:,0]) - 1) / self.period_harmonic * (self.x_length-1)).astype(np.uint32)
            pass
        elif option == 'polar':
            if self.he3:
                ## map one dyad to angle, the other to radius
                def polHe3Tx(periods):
                    norm = self.basis_cents / self.limit
                    theta = norm[:,0] * np.pi * self.period_harmonic
                    radOrder = 1 if "polar-rad-order" not in kwArgs else kwArgs["polar-rad-order"]
                    rad = ((norm[:,1] + 1) ** radOrder - 1) / (2**radOrder - 1)
                    x = np.rint((rad * np.cos(theta) + 1) * 0.5 * (self.x_length - 1)).astype(np.uint32)
                    y = np.rint((rad * np.sin(theta) + 1) * 0.5 * (self.x_length - 1)).astype(np.uint32)
                    return (x, y)
                tx = polHe3Tx
                # def polHe3Mask():
                #     mx,my = self._get_mask_mesh()

            else:
                pass
        elif option == 'odd-split' and self.he3:
            # first set of dyads on a vertical line down the middle
            # next one look at the integer and check if its even or odd go either left or right
            def oddSplitTx(periods):
                norm = self.basis_cents / self.limit
                d2_dist = norm[:, 1] * (self.x_length - 1)
                sign = np.power(-1, (periods[:,2] & 1))
                x = np.rint((d2_dist * sign + self.x_length) * 0.5).astype(np.uint32)
                y =  np.rint(norm[:, 0] * (self.x_length - 1) + d2_dist).astype(np.uint32)
                return (x, y)
            tx = oddSplitTx
            def oddSplitMask():
                mx, my = self._get_mask_mesh()
                return abs(0.5 * self.x_length - my) <= abs(mx * 0.5)
                
            mask_fnc = oddSplitMask
        else:
            self.basis_transform_func = None
            self.basis_transform_option = None
            self.entropy_mask_fnc = None
            raise Exception("Unknown basis transfer option: " + option)

        self.basis_transform_func = tx
        self.basis_transform_option = option
        self.entropy_mask_fnc = mask_fnc

    def _transform_basis(self):
        self._vprint(1, 'Transforming basis...')
        if self.basis_transform_func is None:
            if self.he3:
                self.basis_triad_x = np.round((self.basis_cents[:,0] + (self.basis_cents[:,1] / 2)) / self.res).astype(np.int64)
                self.basis_triad_y = np.round(self.basis_cents[:,1] * self.tri_y_scalar / self.res).astype(np.int64)
                self.basis_transform = (self.basis_triad_y, self.basis_triad_x)
            else: 
                self.basis_transform = np.rint(self.basis_cents / self.res).astype(np.uint32)
        else:
            self.basis_transform = self.basis_transform_func(self.basis_periods)

        self.updateTransform = False
        self.updateDistribution = True
    
    def _distribute_basis(self):
        array = self.basis_distribution
        array_alpha = self.basis_distribution_alpha
        shape = self.basis_distribution.shape
        if shape[0] != self.x_length:
            if shape[0] > self.x_length: # don't resize just take slices
                slices = 2 if self.he3 else 1
                self.basis_distribution = array[tuple(slice(None, self.x_length) for _ in range(slices))]
                self.basis_distribution_alpha =  array_alpha[tuple(slice(None, self.x_length) for _ in range(slices))]
            else:
                newShape = self.x_length
                if self.he3:
                    newShape = (self.x_length, self.x_length)
                self.basis_distribution = np.resize(self.basis_distribution, newShape)
                self.basis_distribution_alpha = np.resize(self.basis_distribution_alpha, newShape)

        self.basis_distribution.fill(0)
        self.basis_distribution_alpha.fill(0)

        np.add.at(self.basis_distribution, self.basis_transform, self.basis_weights)
        np.add.at(self.basis_distribution_alpha, self.basis_transform, self.basis_weight_alphas)

        self.updateDistribution = False

    def _prepare_spread(self):
        if self.updateSpread:
            self.i_ss2 = 1 / (self.s**2 * 2)
            # self.ssqrt2pi = self.s * 2.50662827463

            s_range = np.round(self.s * 5)
            axis = np.arange(-s_range, s_range, 1)
            if not self.he3:
                self.basis_spread = np.exp(-(axis**2) * self.i_ss2)
            else:
                x, y = np.meshgrid(axis, axis)
                self.basis_spread = np.exp(-((x**2 + y**2) * self.i_ss2))
        
        if self.updateSpread or self.updateAlpha:
            self.basis_spread_alphas = self.basis_spread ** self.a

        self.updateSpread = False

    def _do_convolve(self):
        self._vprint(1, "Convolving...")

        psi = signal.convolve(self.basis_distribution, self.basis_spread, 'same')
        pa = signal.convolve(self.basis_distribution_alpha, self.basis_spread_alphas, 'same')

        sigma = 1e-16
        alpha = self.a
        if alpha == 1:
            alpha = sigma

        return np.log((pa + sigma) / ((psi ** alpha) + sigma)) / (1 - alpha)
    
    def _prepare_entropy_mask(self):
        if self.entropy_mask_fnc is not None:
            self._vprint(1, "Preparing mask...")
            self.entropy_mask = self.entropy_mask_fnc()
        self.updateMask = False

    def _mask_triadic_entropy(self, entropy):
        if self.entropy_mask is not None:
            self._vprint(1, "Masking...")
            masked = 7 - entropy
            masked[~self.entropy_mask] = 0
            return masked
        self._vprint(1, "No mask...")
        return 7 - entropy

    # Quickest way to get the data
    def getEntropy(self, loadFile=True):
        if self.he3:
            file = os.path.join(os.path.dirname(__file__), "he_data", "3he_{}.npy".format(self.suffix()))
        else:
            file = os.path.join(os.path.dirname(__file__), "he_data", "he_{}.npy".format(self.suffix()))

        if loadFile and os.path.exists(file): 
            if self.loadedEntropyFile != file:
                self.Entropy = np.load(file)
                self.loadedEntropyFile = file
        else:
            return self.calculate()
        return self.Entropy

    def calculate(self):
        if self.updateX:
            self._prepare_x_axis()
        if self.regenBasis:
            self._generate_basis()
        if self.updateBasis:
            self._prepare_basis_periods()
        if self.updateAlpha:
            self._weight_basis()
            self._prepare_spread()
            self.updateAlpha = False
        if self.updateWeights:
            self._weight_basis()
        if self.updateSpread:
            self._prepare_spread()
        if self.updateTransform:
            self._transform_basis()
        if self.he3 and self.updateMask:
            self._prepare_entropy_mask()
        if self.updateDistribution:
            self._distribute_basis()
            if not self.he3:
                self.Entropy = self._do_convolve()
            else:
                self.Entropy = self._mask_triadic_entropy(self._do_convolve())

            if self.weight_option == "all" :
                self.EntropyAltWeights = {}
                for option in ["lencf", "lenmaxcf", "sumcf"]:
                    self.setWeightingOption("lencf")
                    self.EntropyAltWeights[option] = self._do_convolve()

    def saveEntropy(self, filepath=None):
        if self.he3:
            file = os.path.join(os.path.dirname(__file__), "he_data", "3he_{}".format(self.suffix()))
        else:
            file = os.path.join(os.path.dirname(__file__), "he_data", "he_{}".format(self.suffix()))
        
        np.save(file, self.Entropy)

        if filepath is None:
            filepath = file + ".txt"

        self._vprint(1, f"Writing: {filepath}")
        np.savetxt(filepath, self.Entropy,  fmt="%f")

    def getEntropyPlotData(self, min_cents=None, max_cents=None):
        if self.he3:
            pass # todo
            return (self.x_axis, self.Entropy)

        start = 0
        if min_cents is not None:
            start = max(0, int(np.round(min_cents / self.res)))

        end = self.x_length
        if max_cents is not None:
            end = min(self.x_length, int(np.round(max_cents / self.res)))

        return (self.x_axis[start:end], self.Entropy[start:end])

    def makePlot(self, save=True, show=True, **hePlotArgs):
        self._vprint(1, "Plotting...")

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

        if self.he3:
            plotArgs["imshow"]=True
            plotArgs["figsize"]=(16,16)
            plotArgs["cmap"]="inferno"

        if self.plot:
            self.plot.close()

        self.plot = create_plot(plot_data, 
                  title, 
                  "Dyad (cents)", 
                  "Dissonance",
                  **plotArgs
                  )
        
        if save:
            outdir = "he_plots"
            basename = None
            if self.output_dir:
                (outdir, basename) = os.path.split(os.path.abspath(self.output_dir))
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
            if "filename" in hePlotArgs:
                filename = os.path.join(outdir, hePlotArgs["filename"])
            else:
                # auto name that doesn't overwrite files
                if basename:
                    filename = basename
                else:
                    filename = '{}he_{}'.format(3 if self.he3 else "", self.suffix())
                filename = os.path.join(outdir, filename)
                filename = getUniqueFilename(filename, "png")

            if "dpi" in self.kwArgs:
                dpi = int(self.kwArgs["dpi"])
            else:
                dpi = 480

            self._vprint(1, "Writing: " + filename)
            self.plot.savefig(filename, dpi=dpi, pad_inches=0, bbox_inches='tight')

        if show:
            self.plot.show()

        return self.plot



if __name__ == "__main__": 
    parser = argparse.ArgumentParser("harmonicentro.py")
    parser.add_argument('-n', '--N', type=int, help="Exclusive limit of basis set ratio denominators")
    parser.add_argument('-s', '--spread',type=float, help="Bandwidth of spreading function in cents")
    parser.add_argument('-a', '--alpha', type=float, help="Order of weight scaling")
    parser.add_argument('-r', '--res', type=float, help="Resolution of x-axis in cents")
    parser.add_argument('-l', '--limit', type=float, help="Last cents value to calculate")

    parser.add_argument('-w', '--weight', choices=['default', 'sqrtnd', 'lencf', 'lenmaxcf', 'sumcf', 'all'])
    parser.add_argument('-t', '--tx', choices=['default', 'lin', 'polar', 'odd-split'])

    parser.add_argument('--pl', type=str, help='Comma-separated list of primes to limit by')
    parser.add_argument('--pr', type=str, help='Comma-separated list of primes to reject')
    parser.add_argument('--odd', action='store_true', help='Only include odd-numbered rationals (shortcut for --pr 2)')
    parser.add_argument('--primes', action='store_true', help='Only include strict prime rationals')
    parser.add_argument('--composites', action='store_true', help='Only include strictly composite rationals')

    parser.add_argument('--he3', action='store_true', help='3HE mode')
    parser.add_argument('--plot', action='store_true', help="Display plot")
    parser.add_argument('--ticks', action='store_true', help="Auto-select minima-based x-axis ticks")
    parser.add_argument('--save', action='store_true', help="Save to file")
    parser.add_argument('--no-save', action='store_true', help="Don't save plot file")
    parser.add_argument('--out', type=str, help='Plot output path')
    parser.add_argument('--dpi', type=int, help='Plot output DPI')

    parser.add_argument('--verbose', type=int, help='Print level')

    parsed = parser.parse_args()

    options = vars(parsed)
    saveText = options['save']
    if not saveText:
        savePlot = not options['no_save']
    else:
        savePlot = True

    del options['save']
    plot = options['plot']
    del options['plot']

    heArgs = { k:v for k,v in options.items() if v is not None }
    he = HarmonicEntropy(**heArgs)
    he.calculate()

    if saveText:
        he.saveEntropy()

    if plot:
        he.makePlot(savePlot)
