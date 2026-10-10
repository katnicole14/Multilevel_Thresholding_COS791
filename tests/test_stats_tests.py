import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from experiments.stats_tests import friedman, holm, pairwise_wilcoxon, wilcoxon_summary

ALGORITHMS = ["DE", "JADE", "SHADE", "L-SHADE", "LADE"]


def synthetic_runs(offsets, n_images=3, n_runs=30, seed=0, noise=0.01):
    """One objective, one K; algorithm a scores noise + offsets[a] per (image, run)."""
    rng = np.random.default_rng(seed)
    rows = []
    for image in range(n_images):
        for run in range(n_runs):
            base = rng.normal()
            for algorithm in ALGORITHMS:
                rows.append({"image": f"img{image}", "run": run, "objective": "Otsu", "K": 3,
                             "algorithm": algorithm,
                             "best_fitness": base + offsets.get(algorithm, 0) + rng.normal(0, noise)})
    return pd.DataFrame(rows)


def test_holm_matches_hand_calculation():
    # sorted p: 0.01*3, 0.02*2, 0.04*1 -> 0.03, 0.04, 0.04 (monotone)
    assert holm([0.04, 0.01, 0.02]).tolist() == pytest.approx([0.04, 0.03, 0.04])


def test_holm_caps_at_one():
    assert holm([0.6, 0.7]).tolist() == pytest.approx([1.0, 1.0])


def test_friedman_detects_a_clearly_better_algorithm():
    runs = synthetic_runs({"JADE": 1.0})
    table = friedman(runs, "best_fitness", True, ALGORITHMS)
    assert table.loc[0, "significant"]
    assert table.loc[0, "best"] == "JADE"
    assert table.loc[0, "JADE"] == pytest.approx(1.0)


def test_friedman_identical_algorithms_not_significant():
    runs = synthetic_runs({})
    runs["best_fitness"] = runs.groupby(["image", "run"])["best_fitness"].transform("first")
    table = friedman(runs, "best_fitness", True, ALGORITHMS)
    assert table.loc[0, "p"] == 1.0


def test_lower_is_better_reverses_ranks():
    runs = synthetic_runs({"JADE": -1.0})
    table = friedman(runs, "best_fitness", False, ALGORITHMS)
    assert table.loc[0, "best"] == "JADE"


def test_wilcoxon_winner_and_summary():
    runs = synthetic_runs({"L-SHADE": 1.0}, noise=0.0)
    pairs = pairwise_wilcoxon(runs, "best_fitness", True, ALGORITHMS)
    assert len(pairs) == 10
    lshade = pairs[(pairs["A"] == "L-SHADE") | (pairs["B"] == "L-SHADE")]
    assert (lshade["winner"] == "L-SHADE").all()
    others = pairs[(pairs["A"] != "L-SHADE") & (pairs["B"] != "L-SHADE")]
    assert (others["winner"] == "none").all()

    summary = wilcoxon_summary(pairs, ALGORITHMS)
    assert summary.loc["L-SHADE", "DE"] == "1/0/0"
    assert summary.loc["DE", "L-SHADE"] == "0/0/1"
    assert summary.loc["DE", "JADE"] == "0/1/0"
