#mutation 

"""
mutation.py

This module is fully self-contained: it only needs a population (list/array
of candidate vectors) to run and test against. It does NOT depend on
crossover or selection — it produces the donor vector v_i that crossover
will later combine with the target vector x_i.
"""

import numpy as np


def mutate(population, target_idx, F, bounds=None, repair="clip", rng=None):
    """
    Generate a donor (mutant) vector for one target vector using the
    classic DE/rand/1 strategy:

        v_i = x_r1 + F * (x_r2 - x_r3)

    where r1, r2, r3 are distinct indices, all different from target_idx.

    Parameters
    ----------
    population : array-like, shape (NP, D)
        The current population. NP = population size, D = dimensionality
        (e.g. D = number of thresholds for multilevel thresholding).
    target_idx : int
        Index i of the target vector x_i. r1, r2, r3 are chosen to be
        distinct from this index (and from each other).
    F : float
        Scale factor, typically in (0, 2], most commonly in [0.4, 1.0].
    bounds : tuple(low, high) or array-like of shape (D, 2), optional
        Valid range for each dimension (e.g. (0, 255) for pixel
        thresholds). If None, no boundary repair is applied.
    repair : {"clip", "reflect", "random"}, default "clip"
        Strategy used to fix donor components that fall outside bounds:
          - "clip":    clamp to the nearest bound.
          - "reflect": reflect back into range off the violated bound.
          - "random":  re-sample that component uniformly within bounds.
    rng : numpy.random.Generator, optional
        Random generator to use. If None, a fresh default_rng() is used.
        Pass your own rng for reproducible tests.

    Returns
    -------
    donor : np.ndarray, shape (D,)
        The mutant/donor vector v_i.

    Raises
    ------
    ValueError
        If the population has fewer than 4 members (need target + 3
        distinct others), or target_idx is out of range.
    """
    pop = np.asarray(population, dtype=float)

    if pop.ndim != 2:
        raise ValueError(f"population must be 2D (NP, D), got shape {pop.shape}")

    NP = pop.shape[0]

    if NP < 4:
        raise ValueError(
            f"DE/rand/1 mutation needs at least 4 population members "
            f"(1 target + 3 distinct random), got NP={NP}"
        )

    if not (0 <= target_idx < NP):
        raise ValueError(f"target_idx={target_idx} out of range for NP={NP}")

    if rng is None:
        rng = np.random.default_rng()

    candidate_idxs = [i for i in range(NP) if i != target_idx]
    r1, r2, r3 = rng.choice(candidate_idxs, size=3, replace=False)

    donor = pop[r1] + F * (pop[r2] - pop[r3])

    if bounds is not None:
        donor = _repair(donor, bounds, method=repair, rng=rng)

    return donor


def mutate_population(population, F, bounds=None, repair="clip", rng=None):
    """
    Convenience wrapper: run mutate() for every member of the population,
    producing a full donor population V of the same shape as the input.

    Parameters
    ----------
    population : array-like, shape (NP, D)
    F : float
    bounds : tuple(low, high) or array-like of shape (D, 2), optional
    repair : {"clip", "reflect", "random"}, default "clip"
    rng : numpy.random.Generator, optional

    Returns
    -------
    donors : np.ndarray, shape (NP, D)
    """
    pop = np.asarray(population, dtype=float)
    NP = pop.shape[0]

    if rng is None:
        rng = np.random.default_rng()

    donors = np.empty_like(pop)
    for i in range(NP):
        donors[i] = mutate(pop, i, F, bounds=bounds, repair=repair, rng=rng)

    return donors


def _repair(vector, bounds, method="clip", rng=None):
    """
    Repair a vector that may have components outside the valid bounds.

    Parameters
    ----------
    vector : np.ndarray, shape (D,)
    bounds : tuple(low, high) or array-like of shape (D, 2)
        If a single (low, high) pair is given, it applies to every
        dimension. Otherwise, per-dimension bounds are expected.
    method : {"clip", "reflect", "random"}
    rng : numpy.random.Generator, optional

    Returns
    -------
    repaired : np.ndarray, shape (D,)
    """
    vector = np.array(vector, dtype=float)
    D = vector.shape[0]

    bounds_arr = np.asarray(bounds, dtype=float)
    if bounds_arr.shape == (2,):
        low = np.full(D, bounds_arr[0])
        high = np.full(D, bounds_arr[1])
    elif bounds_arr.shape == (D, 2):
        low = bounds_arr[:, 0]
        high = bounds_arr[:, 1]
    else:
        raise ValueError(
            f"bounds must be shape (2,) or ({D}, 2), got {bounds_arr.shape}"
        )

    if method == "clip":
        return np.clip(vector, low, high)

    elif method == "reflect":
        repaired = vector.copy()
        below = repaired < low
        above = repaired > high
        repaired[below] = 2 * low[below] - repaired[below]
        repaired[above] = 2 * high[above] - repaired[above]
        # in case reflection overshoots the opposite bound, clip as a
        # final safety net
        return np.clip(repaired, low, high)

    elif method == "random":
        if rng is None:
            rng = np.random.default_rng()
        repaired = vector.copy()
        out_of_bounds = (repaired < low) | (repaired > high)
        n_bad = np.count_nonzero(out_of_bounds)
        if n_bad:
            repaired[out_of_bounds] = rng.uniform(
                low[out_of_bounds], high[out_of_bounds]
            )
        return repaired

    else:
        raise ValueError(f"Unknown repair method: {method!r}")