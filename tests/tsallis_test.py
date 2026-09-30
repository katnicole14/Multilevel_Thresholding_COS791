import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "Objective_functions"))

import tsallis as ts


def make_test_histogram():
    """The exact 10-pixel example we hand-calculated: 10,12,15,50,55,60,200,205,210,210"""
    histogram = [0.0] * 256
    for shade in [10, 12, 15, 50, 55, 60, 200, 205]:
        histogram[shade] = 0.1
    histogram[210] = 0.2
    return histogram


def test_tsallis_hand_calculated():
    histogram = make_test_histogram()
    score = ts.tsallis([30, 100], histogram, q=0.8)
    assert math.isclose(score, 4.571, abs_tol=0.01)


def test_tsallis_converges_to_kapur_as_q_approaches_1():
    histogram = make_test_histogram()
    score = ts.tsallis([30, 100], histogram, q=0.99999)
    # Hand-calculated Kapur score on the same thresholds is ~3.238
    assert math.isclose(score, 3.238, abs_tol=0.01)


def test_tsallis_handles_empty_class():
    histogram = make_test_histogram()
    score = ts.tsallis([5, 100], histogram, q=0.8)  # threshold at 5 -> empty first class
    assert math.isfinite(score)


def test_tsallis_single_shade_pile_has_zero_entropy():
    histogram = [0.0] * 256
    histogram[100] = 1.0
    score = ts.tsallis([50, 150], histogram, q=0.8)
    assert math.isclose(score, 0.0, abs_tol=0.001)


def test_tsallis_rewards_even_spread():
    histogram_even = [0.0] * 256
    for shade in [10, 20, 30, 40]:
        histogram_even[shade] = 0.25
    histogram_skewed = [0.0] * 256
    histogram_skewed[10] = 0.85
    for shade in [20, 30, 40]:
        histogram_skewed[shade] = 0.05

    score_even = ts.tsallis([5], histogram_even, q=0.8)
    score_skewed = ts.tsallis([5], histogram_skewed, q=0.8)
    assert score_even > score_skewed
