"""
test_evaluation.py

Tests for the segmentation and image metrics used in Experiment 1, and for
the white-border crop.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from evaluation.metrics import psnr, ssim, uniformity
from evaluation.segmentation import class_labels, segment
from Histogram.Process import crop_white_border


def random_image(seed=0, shape=(60, 80)):
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 255, shape[1])[None, :] + rng.normal(0, 20, shape)
    return np.clip(base, 0, 255).astype(np.uint8)


# --- segmentation ---------------------------------------------------------

def test_class_convention_matches_objectives():
    """Class 0 is [0, t1], class 1 is [t1+1, t2], ..."""
    gray = np.array([[0, 50, 51, 100, 101, 255]], dtype=np.uint8)
    assert class_labels(gray, [50, 100]).tolist() == [[0, 0, 1, 1, 2, 2]]


def test_segment_replaces_pixels_with_class_mean():
    gray = np.array([[10, 20, 200, 210]], dtype=np.uint8)
    assert segment(gray, [100]).tolist() == [[15, 15, 205, 205]]


def test_segment_handles_empty_class():
    gray = np.array([[10, 20, 200, 210]], dtype=np.uint8)
    out = segment(gray, [50, 100, 150])
    assert out.tolist() == [[15, 15, 205, 205]]


# --- PSNR -----------------------------------------------------------------

def test_psnr_identical_images_is_infinite():
    gray = random_image()
    assert psnr(gray, gray) == float("inf")


def test_psnr_known_value():
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.full((10, 10), 10, dtype=np.uint8)  # MSE = 100
    assert psnr(a, b) == pytest.approx(10 * np.log10(255 ** 2 / 100))


def test_psnr_increases_with_more_thresholds():
    gray = random_image(1)
    few = psnr(gray, segment(gray, [85, 170]))
    many = psnr(gray, segment(gray, list(range(20, 250, 20))))
    assert many > few


# --- SSIM -----------------------------------------------------------------

def test_ssim_identical_images_is_one():
    gray = random_image(2)
    assert ssim(gray, gray) == pytest.approx(1.0)


def test_ssim_matches_scikit_image():
    skimage_metrics = pytest.importorskip("skimage.metrics")
    gray = random_image(3)
    seg = segment(gray, [60, 120, 180])
    expected = skimage_metrics.structural_similarity(
        gray, seg, data_range=255, gaussian_weights=True,
        sigma=1.5, use_sample_covariance=False,
    )
    assert ssim(gray, seg) == pytest.approx(expected, abs=1e-10)


# --- Uniformity -----------------------------------------------------------

def test_uniformity_is_one_for_perfectly_uniform_classes():
    gray = np.array([[10, 10, 200, 200]], dtype=np.uint8)
    assert uniformity(gray, [100]) == pytest.approx(1.0)


def test_uniformity_known_value():
    # classes {0, 10} and {200}: within-class SSE = 50, N = 3, range = 200, K = 1
    gray = np.array([[0, 10, 200]], dtype=np.uint8)
    expected = 1 - 2 * 1 * 50 / (3 * 200 ** 2)
    assert uniformity(gray, [100]) == pytest.approx(expected)


def test_uniformity_matches_direct_formula():
    gray = random_image(4)
    t = [50, 100, 150, 200]
    labels = class_labels(gray, t)
    f = gray.astype(float)
    sse = sum(((f[labels == j] - f[labels == j].mean()) ** 2).sum()
              for j in np.unique(labels))
    expected = 1 - 2 * len(t) * sse / (f.size * (f.max() - f.min()) ** 2)
    assert uniformity(gray, t) == pytest.approx(expected)


# --- white border ---------------------------------------------------------

def test_crop_white_border_removes_only_full_white_edges():
    inner = random_image(5, (20, 30))
    inner[0, 0] = 255  # a white pixel inside the photo must stay
    framed = np.full((30, 44), 255, dtype=np.uint8)
    framed[4:24, 7:37] = inner
    assert np.array_equal(crop_white_border(framed), inner)


def test_crop_white_border_without_border_is_unchanged():
    gray = random_image(6)
    assert np.array_equal(crop_white_border(gray), gray)
