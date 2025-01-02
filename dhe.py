from he import HarmonicEntropy, create_plot
import numpy as np
from utils import *

from scipy.signal import convolve, argrelmin

import weighting

tri_y_scalar = np.sqrt(3) / 2
def centsTo3HeYx(cy, cx, res=1):
    return (
        np.round(cy * tri_y_scalar / res).astype(np.int64),
        np.round((cx + (cy / 2)) / res).astype(np.int64)
    )

class DeltaHarmonicEntropy:

    def __init__(self, **kwArgs):

        self.useNewWeighting = True # hack until solving 3HE issue

        self.x_min_hz = 100
        self.x_range = 2
        self.x_res = 1

        self.int_limit = 100
    

        self.updateX = True
        self.updateHe = True
        self.recalculate = True

    def _prepare_x_axis_(self):
        self.x_range_cents = ratio_to_cents(self.x_range)
        self.x_max = self.x_min_hz * self.x_range
        self.x_cents = np.arange(0, int(np.ceil(self.x_range_cents + self.x_res)), step=self.x_res)
        self.x_length = len(self.x_cents)
        self.x_dec = np.exp2(self.x_cents / 1200.0)

        self.he3_limit = ratio_to_cents(self.x_range * 4)

        self.differenceEntropy = np.zeros(self.x_length)
        self.combinationEntropy = np.zeros(self.x_length)

        self.updateX = False
        self.updateHe = True

    def _prepare_he_data_(self):
        self.he2 = HarmonicEntropy(spread=13, N=self.int_limit, limit=self.x_range_cents, res=self.x_res)
        self.he2.calculate()
        self.basis_length = self.he2.basis_length
        self.basis_cents = self.he2.basis_cents

        self.spread = 10
        self.alpha = self.he2.a

        self.he3 = HarmonicEntropy(spread=15, N=self.int_limit, limit=self.he3_limit, he3=True)
        # self.he3.calculate()
        # self.EntropyHe3 = self.he3.getEntropy(True, True)
        # self.he3.saveEntropy()

        self.updateHe = False
        self.recalculate = True

    def doPlotting(self):
        plt = create_plot(self.he2.Entropy, "Difference 3HE (2DHE) over 2HE", 
                plot_dhe=self.differenceEntropy, plot_dhe_label='2DHE',

                # plot_dhe_weights=self.dheWeights, plot_dhe_weights_label='DHE Weights'
                plot_che=self.combinedEntropy, plot_che_label='2HE + 2DHE',

                # plot_2dhe=self.he2.Entropy + self.differenceEntropy, plot_2dhe_label="2HE+DHE",
                # plot_all=self.he2.Entropy + self.differenceEntropy + self.combinationEntropy, plot_all_label="2HE+DHE+CHE"
                )
        

        num_ticks = 11
        
        freq_ticks = np.linspace(1, self.x_range, num_ticks)
        cents_ticks = [ np.round(ratio_to_cents(x)).astype(np.uint64) for x in freq_ticks ]
        x_labels = [ f'f * {x:.1f}' for x in freq_ticks ]
        plt.xticks(cents_ticks, x_labels)

        plt.xlabel(f'Dyad {self.x_min_hz}hz')
        plt.ylabel('HE')
        plt.show()


    # TODO maybe use this to calculate a dedicated 3HE with a limited set of ratios
    # def _prepare_basis_periods(self):
    #     self._vprint(1, f"Preparing basis...")

        # self.basis_dyads = get_farey_sequence_basis(self.int_limit, self.x_range)
        # self.basis_cents = nd_basis_to_cents(self.basis_dyads)

        # if self.he3:
            # self.basis_periods = np.asarray(self.basis_set).copy()
        # else:

        # self.basis_length = self.basis_dyads.shape[0]

        # self.deWeights = np.zeros(self.x_length)
        # self.deWeightsAlpha = np.zeros(self.x_length)
        
        # self.ceWeights = np.zeros(self.x_length)
        # self.ceWeightsAlpha = np.zeros(self.x_length)

        # self.dhe_basis = np.ndarray((self.x_length, 3))
        # self.che_basis = np.ndarray((self.x_length, 3))

        # for i in range(self.basis_length):
            # d,n = self.basis_dyads[i, :]
            # dif = n - d
            # comb = n + d
            # self.dhe_basis[i] = np.asarray([ dif, d, n ]).astype(np.int32)
            # self.che_basis[i] = np.asarray([ d, n, comb ]).astype(np.int32)

    def calculate_new_weighting(self):
        weights = np.zeros(self.x_length)

        for i in range(self.x_length):
            d = self.x_min_hz
            n = d * self.x_dec[i]
            dif = n - d
            comb = n + d

            if dif == 0:
                weights[i] = 1
            else:
                est_weight = weighting.he_decimal_approx([ d / dif, n / d ])
                weights[i] = 1 / np.sqrt(est_weight)


        self.dheWeights = ( 0 - weights )

        s_range = self.spread * 5
        spread_x = np.arange(-s_range, s_range, 1)
        self.dheSpread = np.exp(-(spread_x**2) / (2*self.spread**2))

        psi = convolve(self.dheWeights, self.dheSpread, 'same')
        pa = convolve(self.dheWeights ** self.alpha, self.dheSpread ** self.alpha, 'same')

        sigma = 1e-16
        alpha = self.alpha
        if alpha == 1:
            alpha = sigma

        return np.log((pa + sigma) / ((psi ** alpha) + sigma)) / (1 - alpha)

    # need to figure how to nicely handle super large ranges hard to get 3HE for
    def calculate_from_3he(self):

        # min_octave_cents = np.exp2(np.floor(np.log2(self.he3.limit / 1200.0))) * 1200
        basis_extra_octaves = np.floor(np.log2(self.he3_limit / 1200.0)).astype(np.uint32)

        for i in range(self.x_length):
            d = self.x_min_hz
            n = d * self.x_dec[i]
            dif = n - d
            comb = n + d

            dc1 = ratio_to_cents(d / dif)
            dc2 = self.basis_cents[i]
            og_dc1 = dc1
            
            span = dc1 + dc2
            # cheap OOB interpolation
            if span > self.he3_limit:
                scale = span // self.he3_limit
                # scale = int(dc1 / 1200.0)
                dc1 -= self.he3.limit * scale
                
                # if dc1 + dc2 > self.he_limit:
                #     dc1 -= 1200
                #     scale += 1

                scale *= np.sqrt(2)

            dy,dx = centsTo3HeYx(dc2, dc1)
            # self.differenceEntropy[i] = self.EntropyHe3[dy, dx] + scale
            # self.differenceEntropy[i] = self.EntropyHe3[dy, dx] + scale

            # cc1 = self.basis_cents[i]
            # cc2 = ratio_to_cents(comb / n)
            # scale = 0
            # if cc2 > self.he3_limit:
            #     scale = int(cc2 / 1200.0) - basis_extra_octaves
            #     cc2 = cc2 - 1200.0 * scale
            #     scale = 1.0 / scale

            # cy,cx = centsTo3HeYx(cc2, cc1)
            # self.combinationEntropy[i] = self.EntropyHe3[cy, cx]
            # self.combinationEntropy[i] = self.EntropyHe3[cy, cx]

    def calculate_combine_2he(self):
        return np.sqrt(self.he2.Entropy ** 2 + self.differenceEntropy ** 2)

    def calculate(self):
        if self.updateX:
            self._prepare_x_axis_()
        if self.updateHe:
            self._prepare_he_data_()
        if self.recalculate:
            if self.useNewWeighting:
                self.differenceEntropy = self.calculate_new_weighting()
            else:
                self.differenceEntropy = self.calculate_from_3he()
            
            self.combinedEntropy = self.calculate_combine_2he()
            self.recalculate = False

    def printMinima(self, mode='combined-dhe'):
        if mode == 'combined-dhe':
            data = self.combinedEntropy
        else:
            data = self.differenceEntropy

        minima = argrelmin(data)[0]
        print("Minima:")
        for index in minima:
            cents = dhe.x_cents[index]
            freq = cents_to_ratio(cents) * dhe.x_min_hz
            # print(f'{index} ({dhe.combinedEntropy[index]:.3f}): ({freq-dhe.x_min_hz:.3f}) {dhe.x_min_hz:.3f} - {freq:.3f}')
            print(f'[{freq-dhe.x_min_hz:.2f}hz {dhe.x_min_hz:.2f}hz {freq:.2f}hz]')
    

if __name__ == '__main__':

    dhe = DeltaHarmonicEntropy()
    dhe.calculate()
    dhe.doPlotting()



