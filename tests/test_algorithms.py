"""
test_algorithms.py

Checks shared by all five DE variants, which use the same interface:
    run(initial_population, objective_function, seed, maximum_function_evaluations) -> dict
"""

import sys
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from algorithms.jade import run_jade
from algorithms.late_acceptance import run_lade
from algorithms.SelectMainLP import run_de
from algorithms.Shade import run_lshade, run_shade
from Objective_functions.Kapur import kapur_entropy
from Objective_functions.otsu import otsu_fitness, precompute_cumulative
from Objective_functions.tsallis import tsallis

ALGORITHMS = {
    "DE": run_de,
    "JADE": run_jade,
    "SHADE": run_shade,
    "L-SHADE": run_lshade,
    "LADE": run_lade,
}


def bimodal_histogram():
    """Synthetic image histogram with several peaks, so no image files are needed."""
    levels = np.arange(256)
    hist = sum(
        np.exp(-0.5 * ((levels - centre) / 12) ** 2) * weight
        for centre, weight in [(40, 3), (100, 2), (160, 4), (220, 1)]
    )
    hist = np.round(hist * 1000) + 1
    return hist, hist / hist.sum()


HIST, PDF = bimodal_histogram()
P, S = precompute_cumulative(PDF)
OBJECTIVES = {
    "Otsu": partial(otsu_fitness, P=P, S=S),
    "Kapur": partial(kapur_entropy, probabilities=PDF),
    "Tsallis": lambda t: tsallis(t, HIST, 0.8),
}


def initial_population(seed, size=20, k=5):
    return np.random.default_rng(seed).uniform(1, 254, size=(size, k))


@pytest.mark.parametrize("objective_name", OBJECTIVES)
@pytest.mark.parametrize("algorithm_name", ALGORITHMS)
def test_returns_valid_thresholds(algorithm_name, objective_name):
    result = ALGORITHMS[algorithm_name](
        initial_population(1), OBJECTIVES[objective_name], 1, 400
    )
    t = result["best_thresholds"]
    assert len(t) == 5
    assert all(isinstance(v, int) for v in t)
    assert all(a < b for a, b in zip(t, t[1:]))
    assert 1 <= t[0] and t[-1] <= 254
    assert np.isclose(result["best_fitness"], OBJECTIVES[objective_name](np.array(t)))


@pytest.mark.parametrize("algorithm_name", ALGORITHMS)
def test_uses_exact_function_evaluation_budget(algorithm_name):
    calls = []

    def counting(t):
        calls.append(1)
        return OBJECTIVES["Otsu"](t)

    result = ALGORITHMS[algorithm_name](initial_population(2), counting, 2, 517)
    assert result["function_evaluations"] == 517
    assert len(calls) == 517


@pytest.mark.parametrize("algorithm_name", ALGORITHMS)
def test_same_seed_is_reproducible(algorithm_name):
    run = ALGORITHMS[algorithm_name]
    a = run(initial_population(3), OBJECTIVES["Otsu"], 3, 600)
    b = run(initial_population(3), OBJECTIVES["Otsu"], 3, 600)
    assert a["best_thresholds"] == b["best_thresholds"]
    assert a["convergence_history"] == b["convergence_history"]


@pytest.mark.parametrize("algorithm_name", ALGORITHMS)
def test_convergence_history_is_non_decreasing(algorithm_name):
    result = ALGORITHMS[algorithm_name](initial_population(4), OBJECTIVES["Otsu"], 4, 1000)
    fes = [f for f, _ in result["convergence_history"]]
    best = [b for _, b in result["convergence_history"]]
    assert fes[0] == 20 and fes[-1] == 1000
    assert all(later >= earlier for earlier, later in zip(best, best[1:]))


@pytest.mark.parametrize("algorithm_name", ALGORITHMS)
def test_improves_on_initial_population(algorithm_name):
    objective = OBJECTIVES["Otsu"]
    pop = initial_population(5)
    initial_best = max(objective(row) for row in pop)
    result = ALGORITHMS[algorithm_name](pop, objective, 5, 2000)
    assert result["best_fitness"] > initial_best


def test_lshade_population_shrinks_to_minimum():
    result = run_lshade(initial_population(6, size=40), OBJECTIVES["Otsu"], 6, 2000)
    assert result["final_population_size"] == 4


def test_shade_archive_is_used_and_capped():
    result = run_shade(initial_population(7), OBJECTIVES["Otsu"], 7, 2000)
    assert 0 < result["archive_size"] <= 20


@pytest.mark.parametrize("algorithm_name", ALGORITHMS)
def test_kapur_with_empty_classes_keeps_parameters_finite(algorithm_name):
    """Kapur gives -inf for empty classes; adaptation must not turn into NaN."""
    # Only three bands are occupied, so many random thresholds give empty classes.
    sparse_pdf = np.zeros(256)
    for start, stop in [(20, 60), (100, 140), (200, 230)]:
        sparse_pdf[start:stop] = 1.0
    sparse_pdf /= sparse_pdf.sum()
    objective = partial(kapur_entropy, probabilities=sparse_pdf)
    with np.errstate(all="raise"):
        result = ALGORITHMS[algorithm_name](initial_population(8), objective, 8, 1000)
    assert np.isfinite(result["best_fitness"])
