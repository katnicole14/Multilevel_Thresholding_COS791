"""
test_late_acceptance.py

Standalone tests for late_acceptance.py (Late Acceptance DE). Uses simple
hand-verifiable fitness functions so these tests don't depend on real
image data, plus one smoke test on a real image with Otsu.
"""

import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from algorithms.late_acceptance import late_acceptance_select, run_lade

TARGET = np.array([64, 128, 192])


def distance_to_target(thresholds):
    """Toy fitness: 0 at TARGET, more negative the further away."""
    return -float(np.sum(np.abs(np.asarray(thresholds) - TARGET)))


def random_population(seed, population_size=20, k=3):
    rng = np.random.default_rng(seed)
    return rng.uniform(1, 254, size=(population_size, k))


# --- selection rule -------------------------------------------------------

def test_select_accepts_improvement_over_target():
    assert late_acceptance_select(5.0, 4.0, 10.0)


def test_select_accepts_tie_with_target():
    assert late_acceptance_select(4.0, 4.0, 10.0)


def test_select_accepts_worse_than_target_but_better_than_history():
    assert late_acceptance_select(3.0, 4.0, 2.0)


def test_select_rejects_worse_than_both():
    assert not late_acceptance_select(1.0, 4.0, 2.0)


# --- input validation -----------------------------------------------------

def test_raises_on_non_2d_population():
    with pytest.raises(ValueError):
        run_lade([10, 20, 30], distance_to_target, 0, 100)


def test_raises_on_small_population():
    with pytest.raises(ValueError):
        run_lade(random_population(0, 3), distance_to_target, 0, 100)


def test_raises_on_bad_history_length():
    with pytest.raises(ValueError):
        run_lade(random_population(0), distance_to_target, 0, 100, history_length=0)


def test_raises_when_budget_below_population():
    with pytest.raises(ValueError):
        run_lade(random_population(0), distance_to_target, 0, 10)


# --- behaviour ------------------------------------------------------------

def test_respects_function_evaluation_budget():
    calls = []

    def counting(thresholds):
        calls.append(1)
        return distance_to_target(thresholds)

    result = run_lade(random_population(1), counting, 1, 517)
    assert result["function_evaluations"] == 517
    assert len(calls) == 517


def test_same_seed_is_reproducible():
    a = run_lade(random_population(2), distance_to_target, 7, 2000)
    b = run_lade(random_population(2), distance_to_target, 7, 2000)
    assert a["best_thresholds"] == b["best_thresholds"]
    assert a["convergence_history"] == b["convergence_history"]


def test_best_thresholds_are_valid():
    result = run_lade(random_population(3, k=5), lambda t: -float(np.sum(t)), 3, 1000)
    t = np.array(result["best_thresholds"])
    assert len(t) == 5
    assert np.all(np.diff(t) > 0)
    assert t[0] >= 1 and t[-1] <= 254


def test_convergence_is_monotonically_non_decreasing():
    result = run_lade(random_population(4), distance_to_target, 4, 3000)
    best = [f for _, f in result["convergence_history"]]
    assert all(later >= earlier for earlier, later in zip(best, best[1:]))


def test_search_finds_optimum():
    result = run_lade(random_population(5), distance_to_target, 5, 6000)
    assert result["best_fitness"] >= -3


def test_history_length_one_is_greedy_de():
    """With L=1 the history only holds f(x_i), so nothing worse is accepted."""
    result = run_lade(
        random_population(6), distance_to_target, 6, 3000, history_length=1
    )
    assert result["late_acceptances"] == 0


def test_longer_history_accepts_worse_trials():
    result = run_lade(
        random_population(6), distance_to_target, 6, 3000, history_length=20
    )
    assert result["late_acceptances"] > 0


def test_runs_with_otsu_on_real_image():
    image_path = Path(__file__).resolve().parent.parent / "BDS500" / "img1.png"
    if not image_path.exists():
        pytest.skip("BDS500 test image not available")

    from Histogram.Process import build_histogram, load_grayscale
    from Objective_functions.otsu import otsu_fitness, precompute_cumulative

    _, pdf = build_histogram(load_grayscale(image_path))
    P, S = precompute_cumulative(pdf)
    objective = partial(otsu_fitness, P=P, S=S)

    result = run_lade(random_population(8, k=3), objective, 8, 2000)
    assert len(result["best_thresholds"]) == 3
    assert result["best_fitness"] > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
