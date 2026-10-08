"""
metrics.py

Image reconstruction metrics for Experiment 1 (all: higher is better).

PSNR  Peak Signal-to-Noise Ratio between the original and segmented image.
SSIM  Structural Similarity Index (Wang et al., 2004), Gaussian 11x11 window,
      sigma = 1.5, K1 = 0.01, K2 = 0.03, as in the original paper.
U     Uniformity measure (Levine & Nazif, 1985; Sahoo et al., 1988):
      how homogeneous the grey levels are inside each class.

References
----------
Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
quality assessment: from error visibility to structural similarity. IEEE
Transactions on Image Processing, 13(4), 600-612.
Sahoo, P. K., Soltani, S., & Wong, A. K. C. (1988). A survey of thresholding
techniques. Computer Vision, Graphics, and Image Processing, 41(2), 233-260.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

try:
    from .segmentation import class_labels
except ImportError:
    from segmentation import class_labels


def psnr(original, segmented, data_range=255.0):
    """PSNR = 10 log10(MAX^2 / MSE) in dB. Identical images give +inf."""
    mse = np.mean((original.astype(float) - segmented.astype(float)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(data_range ** 2 / mse))


def ssim(original, segmented, data_range=255.0, sigma=1.5, truncate=3.5):
    """
    Mean SSIM over the image (Wang et al., 2004):

        SSIM = ((2 mu_x mu_y + C1)(2 sigma_xy + C2)) /
               ((mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2))

    with local statistics from a Gaussian window (sigma 1.5, 11x11) and
    C1 = (0.01 L)^2, C2 = (0.03 L)^2. Border pixels closer than the window
    radius are excluded from the mean. Matches
    skimage.metrics.structural_similarity(..., gaussian_weights=True,
    sigma=1.5, use_sample_covariance=False, data_range=255).
    """
    x = original.astype(np.float64)
    y = segmented.astype(np.float64)

    def blur(image):
        return gaussian_filter(image, sigma=sigma, truncate=truncate, mode="reflect")

    mu_x, mu_y = blur(x), blur(y)
    var_x = blur(x * x) - mu_x ** 2
    var_y = blur(y * y) - mu_y ** 2
    cov_xy = blur(x * y) - mu_x * mu_y

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    )

    pad = int(truncate * sigma + 0.5)  # window radius (5 for sigma = 1.5)
    return float(ssim_map[pad:-pad, pad:-pad].mean())


 
def uniformity(gray, thresholds):
    """
    U = 1 - 2K * sum_j sum_{i in R_j} (f_i - mu_j)^2 / (N * (f_max - f_min)^2)

    K = number of thresholds, R_j = pixels of class j, mu_j = their mean,
    N = number of pixels, f_max / f_min = brightest / darkest pixel.
    U is 1 for perfectly uniform classes and decreases as classes get
    more varied.
    """
    f = gray.astype(np.float64).ravel()
    labels = class_labels(gray, thresholds).ravel()
    n_classes = len(thresholds) + 1

    counts = np.bincount(labels, minlength=n_classes)
    sums = np.bincount(labels, weights=f, minlength=n_classes)
    sums_sq = np.bincount(labels, weights=f * f, minlength=n_classes)
    # sum over class of (f - mu)^2 = sum f^2 - (sum f)^2 / n
    within = sums_sq - np.divide(sums ** 2, counts, out=np.zeros_like(sums), where=counts > 0)

    spread = (f.max() - f.min()) ** 2
    if spread == 0:
        return 1.0
    return float(1 - 2 * len(thresholds) * within.sum() / (f.size * spread))