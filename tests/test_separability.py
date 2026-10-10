import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from evaluation.separability import class_separability
from Objective_functions.otsu import otsu_fitness, precompute_cumulative


def random_pdf(seed=0):
    hist = np.random.default_rng(seed).integers(0, 100, 256).astype(float)
    return hist / hist.sum()


def test_equals_otsu_variance_over_total_variance():
    pdf = random_pdf(1)
    P, S = precompute_cumulative(pdf)
    total_variance = np.sum((np.arange(256) - S[-1]) ** 2 * pdf)
    for t in ([50], [40, 120, 200], list(range(20, 250, 20))):
        expected = otsu_fitness(np.array(t), P, S) / total_variance
        assert class_separability(t, pdf) == pytest.approx(expected)


def test_is_between_zero_and_one():
    pdf = random_pdf(2)
    for t in ([10], [100, 150], [5, 60, 61, 200]):
        assert 0 <= class_separability(t, pdf) <= 1


def test_is_one_when_every_class_has_a_single_grey_level():
    pdf = np.zeros(256)
    pdf[[30, 120, 220]] = [0.2, 0.5, 0.3]
    assert class_separability([50, 150], pdf) == pytest.approx(1.0)


def test_grows_with_more_thresholds():
    pdf = random_pdf(3)
    assert class_separability([64, 128, 192], pdf) > class_separability([128], pdf)


def test_constant_image_is_zero():
    pdf = np.zeros(256)
    pdf[100] = 1.0
    assert class_separability([50, 150], pdf) == 0.0
