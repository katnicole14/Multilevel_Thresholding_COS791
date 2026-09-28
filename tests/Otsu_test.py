from Histogram import Process
from Objective_functions import Otsu

from skimage.filters import threshold_otsu , threshold_multiotsu
from pathlib import Path # Add missing import for Path
import numpy as np
import itertools

if __name__ == '__main__':
    # Quick sanity check over the BDS500 folder

    #Checking if the Ostu implementation works
    # Use pathlib.Path for consistency with load_grayscale type hint

    gray = Process.load_grayscale(Path("tests/BDS500/img1.png"))
    hist, pdf = Process.build_histogram(gray)
    P, S = Otsu.precompute_cumulative(pdf)

    

for K in [1, 2, 3, 5]:          # drop 7, 9, 11, 12 for testing purposes
    ref_ts = np.sort(threshold_multiotsu(gray, classes=K + 1))
    ref_val = Otsu.otsu_fitness(ref_ts, P, S)
    print(f"\n[K={K}] skimage = {list(ref_ts)}  fitness = {ref_val:.6f}")

    # Full brute force only when it's cheap
    if K <= 2:
        best_val, best_ts = -1, None
        for ts in itertools.combinations(range(1, 255), K):
            v = Otsu.otsu_fitness(np.array(ts), P, S)
            if v > best_val:
                best_val, best_ts = v, ts
        print(f"[K={K}] brute force = {best_ts}  fitness = {best_val:.6f}")

    # Local check: nudge each threshold by ±1..3, nothing should beat skimage
    better = False
    for i in range(K):
        for d in (-3, -2, -1, 1, 2, 3):
            t = ref_ts.copy()
            t[i] += d
            if np.all(np.diff(t) > 0) and t[0] >= 1 and t[-1] <= 254:
                if Otsu.otsu_fitness(t, P, S) > ref_val + 1e-9:
                    better = True
    print(f"[K={K}] any nearby thresholds beat skimage? {better}")   # expect False