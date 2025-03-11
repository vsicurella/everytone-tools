import math
import bisect

from utils import *
import primes

def create_burst(scale, period_cents=1200, theta_scale=2, radius_denominator=0, flip_radius=False):
    max_edo = 1
    for s in scale:
        if s[1] > max_edo:
            max_edo = s[1]
            continue

    if radius_denominator == 0:
        radius_denominator = max_edo

    # rad_scale = math.log2(max_edo)

    theta = [ EaseIoSlope(s[0] / period_cents, theta_scale) * (math.pi * 2) for s in scale ]

    if flip_radius:
        mag_distances = [ (s[1] / radius_denominator) + 1e-4 for s in scale ]
    else:
        mag_distances = [ 1 - (s[1] / radius_denominator) + 1e-4 for s in scale ]

    mag = [ math.log2(2 - d) for d in mag_distances ]

    return (theta, mag)

def et_burst_scale(min_edo=1, max_edo=31, primeFilters=False, primesOnly=True, primeList=[2,3,5,7,11,13,17,19,23,29,31]):
    octave_cents = 1200
    cents = [(0, 1), (octave_cents, 1)]

    # primeList.sort()
    # max_composite_prime = primeList[-1]

    for edo in range(min_edo, max_edo + 1):
        step = octave_cents / edo

        # if primeFilters:
        #     if primesOnly and not primes.is_prime(edo):
        #         continue
            
            # skip = True
            # if edo > max_composite_prime:
            #     for primeEdo in primeList:
            #         den = math.gcd(edo, primeEdo)
            #         if den > 1:
            #             skip = False
            #             break
                
            # if skip:
            #     continue

        for n in range(1, edo, 1):
            den = math.gcd(n, edo)
            if den > 1:
                continue
            # cents.append(step * n)
            bisect.insort(cents, (step * n, edo))
    
    return cents

def temperament_burst(generator=696.77, period=1200.0, size=7, max_edo=313, mode=0):
    g_2 = generator / period
    cf = get_cf(g_2)
    edo_gens = [(1,1)]
    for i in range(2, len(cf)):
        (gen, edo) = get_convergent(cf, i)
        if edo > max_edo:
            break
        edo_gens.append((gen, edo))

    burst = [ (0, 1), (period, 1) ]
    # dbg = {}
    # create one for every edo that supports this temperament
    for (gen,edo) in edo_gens:
        # dbg_edo = []
        gen_cents = period / edo * gen
        i = mode
        tones = 0
        notes_to_add = min(size, edo)
        while tones < notes_to_add:
            tones += 1
            
            cents = round((gen_cents * (tones + i)) % period, 6)
            edo_degree = (gen * (tones + i)) % edo
            
            min_edo = edo
            gen_mult = math.gcd(edo_degree, edo)
            if gen_mult > 1:
                min_edo = edo / gen_mult

            i += 1
            if cents == 0:
                continue
            
            bisect.insort(burst, (cents, min_edo))
    #         dbg_edo.append((cents, min_edo))
    #     dbg[edo] = dbg_edo

    # print(burst)
    # print(dbg)
    return burst


def make_burst_gradient(resolution=128, period_cents=1200, cents_window=1, max_edo=3130, optimized=True):

    grid = []
    tau = math.pi * 2
    pi_2 = math.pi / 2

    half_res = resolution // 2

    mag_thresh = 1e-8

    overlap = resolution % 2

    doSymOptimize = optimized

    for row in range(resolution):
        y = row / half_res
        # ysq = y**2
        y_origin = y - 1
        ysq = y_origin**2

        row = []
        # dbg = []

        # only calculate half due to symmetry
        colLength = resolution
        if doSymOptimize:
            colLength = half_res + 1

        for col in range(colLength):
            x = col / half_res
            x_origin = x - 1

            mag = math.sqrt(ysq + x_origin**2)

            if mag > 1:
                row.append(0)
                continue

            ang = math.atan2(y_origin, x_origin) + pi_2

            cents = (ang % tau) / tau * period_cents

            (edo, degree, error) = get_edo_approx(cents, cents_window, max_edo)

            edo_scalar = edo / max_edo 

            mag_cf = get_cf(mag, 100)
            mag_error = 0
            mag_edo = 1
            for i in range(1, len(mag_cf)):
                (num, den) = get_convergent(mag_cf, i)
                if den > max_edo:
                    break

                mag_edo = den
                approx = float(num) / mag_edo
                percent = approx / mag
                mag_error = 1 - percent
                if abs(mag_error) <= mag_thresh:
                    break
            
            mag_edo_distance = math.log(1 + abs(mag_edo - edo) / max_edo)

            out = (1.0 - edo_scalar) / (mag_edo_distance + 1)
            # out = (1.0 - edo_scalar)

            row.append(out)
            # dbg.append(cents)

        if doSymOptimize:
            for col in range(overlap, half_res):
                # row.append(0)
                row.append(row[half_res - col - 1])

        # print(dbg)
        grid.append(row)

    return grid
