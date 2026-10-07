import numpy as np


def repair_thresholds(t, L=256):
    """
    Convert a raw DE vector into sorted, unique integer thresholds in
    [1, L-2]. Always returns exactly len(t) thresholds, so population
    rows keep a fixed size. This is the single repair function shared by
    the objective functions and all DE variants.
    """
    t = np.sort(np.clip(np.round(t), 1, L - 2).astype(int))
    for i in range(1, len(t)):
        t[i] = max(t[i], t[i - 1] + 1)
    for i in range(len(t) - 1, -1, -1):
        t[i] = min(t[i], L - 2 - (len(t) - 1 - i))
    return t