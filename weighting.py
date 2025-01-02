import numpy as np
import utils

# Assumes ratio in row, column form as d, n

def he_default(ratio):
    return np.sqrt(np.prod(ratio))

def he_n_root(ratio):
    unique = np.unique(ratio)
    return np.power(np.prod(unique), 1 / unique.size)

def he_decimal_approx(decimal_list, int_limit:int=256, tolerance:float=0.6):
    (harmonics, errors) = utils.approximate_ratio(decimal_list, int_limit, tolerance)

    # TODO factor in errors

    return he_default(harmonics)


if __name__ == '__main__':

    from random import Random

    rnd = Random()

    chord_ratios = []
    weights = []
    times = 2
    for i in range(times):
        ratio = rnd.random() + 1
        # weight = cf_distance([ratio])
        
        chord_ratios.append(ratio)
        # weights.append(weight)
    
    chord_ratios = [ 1.2501, 1.201 ]
    chord_weight = he_decimal_approx(chord_ratios)
    chord_weight2 = he_decimal_approx(chord_ratios, tolerance=5)

    print(f"Estimated harmonic weighting: {chord_weight}, adj: {chord_weight2}")
