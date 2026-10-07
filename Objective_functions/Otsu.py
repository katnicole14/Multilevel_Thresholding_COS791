
"""
#Implementing the Otsu
    Definition
1. It is a technique based of between-class- variance (non-parametric threshold selection) concept and it identifies the finiest threshold by maximizing the objective value.
"""
import numpy as np


#Calculate the cumulative sum Pi{k}=sum(pi) for i in class k
#p(i)= h(i)/(M ×N)
def precompute_cumulative(pdf: np.ndarray):
    """
    Precompute the cumulative zeroth-order (P) and first-order (S)
    moments of the histogram. Call this ONCE per image.

    P[i] = sum_{j=0}^{i} p_j        (cumulative probability)
    S[i] = sum_{j=0}^{i} j * p_j    (cumulative intensity-weighted prob)
    """
    levels = np.arange(len(pdf))
    P = np.cumsum(pdf)
    S = np.cumsum(levels * pdf)
    return P, S

def repair_thresholds(t: np.ndarray, L: int = 256) -> np.ndarray:
    """
    Convert a raw DE candidate vector into a valid set of EXACTLY K
    strictly increasing integer thresholds in [1, L-2].
 
    DE mutation/crossover produces continuous, unordered, possibly
    out-of-bounds vectors, so we: round -> clip -> sort -> then push
    any colliding thresholds apart so K is preserved.
    """
    K = len(t)                                   # remember original K FIRST
    t = np.clip(np.round(t).astype(int), 1, L - 2)
    t = np.sort(t)
 
    # forward pass: enforce strictly increasing
    for i in range(1, K):
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] + 1
 
    # if that pushed the top past L-2, fix with a backward pass
    if t[-1] > L - 2:
        t[-1] = L - 2
        for i in range(K - 2, -1, -1):
            if t[i] >= t[i + 1]:
                t[i] = t[i + 1] - 1
    return t


def otsu_fitness(t: np.ndarray, P: np.ndarray, S: np.ndarray, L: int = 256) -> float:
    """
    Compute the between-class variance (sigma_B^2) for a candidate
    threshold vector. This is the value DE should MAXIMIZE.

    t : raw candidate vector from the optimiser (any real values)
    P, S : precomputed cumulative moments from precompute_cumulative()
    """
    thresholds = repair_thresholds(t, L)

    # class boundaries: [0, t1], [t1+1, t2], ..., [tK+1, L-1]
    bounds = np.concatenate(([-1], thresholds, [L - 1]))

    mu_T = S[-1]  # global mean (cumulative sum up to L-1)

    sigma_b2 = 0.0
    for k in range(len(bounds) - 1):
        lo, hi = bounds[k], bounds[k + 1]
        P_lo = P[lo] if lo >= 0 else 0.0
        S_lo = S[lo] if lo >= 0 else 0.0

        omega_k = P[hi] - P_lo
        if omega_k <= 1e-12:
            continue  # empty class, contributes 0 (shouldn't happen post-repair)

        class_sum = S[hi] - S_lo
        mu_k = class_sum / omega_k
        sigma_b2 += omega_k * (mu_k - mu_T) ** 2

    return sigma_b2


"""
#Implementing the Otsu
    Definition
1. It is a technique based of between-class- variance (non-parametric threshold selection) concept and it identifies the finiest threshold by maximizing the objective value.
"""
import numpy as np


#Calculate the cumulative sum Pi{k}=sum(pi) for i in class k
#p(i)= h(i)/(M ×N)
def precompute_cumulative(pdf: np.ndarray):
    """
    Precompute the cumulative zeroth-order (P) and first-order (S)
    moments of the histogram. Call this ONCE per image.

    P[i] = sum_{j=0}^{i} p_j        (cumulative probability)
    S[i] = sum_{j=0}^{i} j * p_j    (cumulative intensity-weighted prob)
    """
    levels = np.arange(len(pdf))
    P = np.cumsum(pdf)
    S = np.cumsum(levels * pdf)
    return P, S

def repair_thresholds(t: np.ndarray, L: int = 256) -> np.ndarray:
    """
    Convert a raw DE candidate vector into a valid set of EXACTLY K
    strictly increasing integer thresholds in [1, L-2].
 
    DE mutation/crossover produces continuous, unordered, possibly
    out-of-bounds vectors, so we: round -> clip -> sort -> then push
    any colliding thresholds apart so K is preserved.
    """
    K = len(t)                                   # remember original K FIRST
    t = np.clip(np.round(t).astype(int), 1, L - 2)
    t = np.sort(t)
 
    # forward pass: enforce strictly increasing
    for i in range(1, K):
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] + 1
 
    # if that pushed the top past L-2, fix with a backward pass
    if t[-1] > L - 2:
        t[-1] = L - 2
        for i in range(K - 2, -1, -1):
            if t[i] >= t[i + 1]:
                t[i] = t[i + 1] - 1
    return t


def otsu_fitness(t: np.ndarray, P: np.ndarray, S: np.ndarray, L: int = 256) -> float:
    """
    Compute the between-class variance (sigma_B^2) for a candidate
    threshold vector. This is the value DE should MAXIMIZE.

    t : raw candidate vector from the optimiser (any real values)
    P, S : precomputed cumulative moments from precompute_cumulative()
    """
    thresholds = repair_thresholds(t, L)

    # class boundaries: [0, t1], [t1+1, t2], ..., [tK+1, L-1]
    bounds = np.concatenate(([-1], thresholds, [L - 1]))

    mu_T = S[-1]  # global mean (cumulative sum up to L-1)

    sigma_b2 = 0.0
    for k in range(len(bounds) - 1):
        lo, hi = bounds[k], bounds[k + 1]
        P_lo = P[lo] if lo >= 0 else 0.0
        S_lo = S[lo] if lo >= 0 else 0.0

        omega_k = P[hi] - P_lo
        if omega_k <= 1e-12:
            continue  # empty class, contributes 0 (shouldn't happen post-repair)

        class_sum = S[hi] - S_lo
        mu_k = class_sum / omega_k
        sigma_b2 += omega_k * (mu_k - mu_T) ** 2

    return sigma_b2

