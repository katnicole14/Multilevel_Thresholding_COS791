"""
segmentation.py

Turn a threshold vector into a segmented image.

K thresholds t1 < t2 < ... < tK split the grey levels into K+1 classes
    C0 = [0, t1], C1 = [t1+1, t2], ..., CK = [tK+1, 255]
(the same class convention the Otsu, Kapur and Tsallis objectives use).
"""

import numpy as np


def class_labels(gray, thresholds):
    """Class index 0..K of every pixel."""
    return np.digitize(gray, np.asarray(thresholds) + 1)


def segment(gray, thresholds):
    """
    Segmented image in which every pixel is replaced by the mean grey level
    of its class. This is the reconstruction PSNR / SSIM are computed on.
    """
    labels = class_labels(gray, thresholds)
    counts = np.bincount(labels.ravel(), minlength=len(thresholds) + 1)
    sums = np.bincount(labels.ravel(), weights=gray.ravel().astype(float),
                       minlength=len(thresholds) + 1)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return np.round(means[labels]).astype(np.uint8)
