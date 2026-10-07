"""
test_late_acceptance.py

Standalone tests for late_acceptance.py. Uses simple hand-verifiable
fitness functions so these tests don't depend on real image data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from algorithms.late_acceptance import late_acceptance_hill_climbing, _perturb


def peak_at_128(thresholds):
    """Toy fitness: higher the closer the single threshold is to 128."""
    return -abs(int(thresholds[0]) - 128)


def test_raises_on_empty_initial_solution():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        late_acceptance_hill_climbing([], peak_at_128, 5, 10, rng)


def test_raises_on_bad_history_length():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        late_acceptance_hill_climbing([50], peak_at_128, 0, 10, rng)


def test_raises_on_negative_iterations():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        late_acceptance_hill_climbing([50], peak_at_128, 5, -1, rng)


def test_zero_iterations_returns_initial_solution():
    rng = np.random.default_rng(0)
    best_solution, best_fitness, convergence = late_acceptance_hill_climbing(
        [50], peak_at_128, history_length=5, max_iterations=0,
        random_generator=rng,
    )
    assert list(best_solution) == [50]
    assert best_fitness == peak_at_128([50])
    assert convergence == []


def test_convergence_is_monotonically_non_decreasing_when_maximizing():
    """best-so-far fitness can only improve or stay flat each iteration."""
    rng = np.random.default_rng(1)
    _, _, convergence = late_acceptance_hill_climbing(
        [10], peak_at_128, history_length=10, max_iterations=200,
        random_generator=rng, levels=256, step_size=20,
    )
    for earlier, later in zip(convergence, convergence[1:]):
        assert later >= earlier


def test_search_improves_toward_optimum():
    """Starting far from the optimum (128), the search should get closer."""
    rng = np.random.default_rng(2)
    best_solution, best_fitness, _ = late_acceptance_hill_climbing(
        [5], peak_at_128, history_length=10, max_iterations=500,
        random_generator=rng, levels=256, step_size=15,
    )
    assert best_fitness >= peak_at_128([5])
    assert abs(int(best_solution[0]) - 128) < abs(5 - 128)


def test_best_solution_length_matches_input():
    rng = np.random.default_rng(3)

    def multi_fitness(thresholds):
        return -float(np.sum(np.abs(np.array(thresholds) - np.array([64, 128, 192]))))

    best_solution, _, _ = late_acceptance_hill_climbing(
        [10, 20, 30], multi_fitness, history_length=5, max_iterations=100,
        random_generator=rng,
    )
    assert len(best_solution) == 3


def test_perturb_stays_within_bounds():
    rng = np.random.default_rng(4)
    solution = np.array([1, 254])
    for _ in range(50):
        neighbour = _perturb(solution, levels=256, random_generator=rng, step_size=10)
        assert np.all(neighbour >= 1) and np.all(neighbour <= 254)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
