"""

Class Separability  eta = sigma_B^2 / sigma_T^2   (Experiment 2 metric).
 
Self-contained: needs only numpy and the normalised histogram `pdf`,
so it can be dropped into the repo (e.g. evaluation/metrics.py) without
depending on any objective-function module.
 
sigma_B^2 : between-class variance of the K+1 classes the thresholds define
sigma_T^2 : total variance of the image histogram
eta is in [0, 1]; closer to 1 = classes are better separated.
Classes follow the same convention as Otsu: [0,t1], [t1+1,t2], ..., [tK+1,L-1].
"""
import numpy as np
 
 
def class_separability(thresholds, pdf) -> float:
    pdf = np.asarray(pdf, dtype=float)
    levels = np.arange(len(pdf))
 
    mu_T = np.sum(levels * pdf)
    sigma_T2 = np.sum((levels - mu_T) ** 2 * pdf)
    if sigma_T2 <= 0:                      # constant image
        return 0.0
 
    t = np.sort(np.asarray(thresholds, dtype=int))
    edges = np.concatenate(([0], t + 1, [len(pdf)]))
 
    sigma_B2 = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        w = pdf[lo:hi].sum()
        if w > 0:
            mu_k = np.sum(levels[lo:hi] * pdf[lo:hi]) / w
            sigma_B2 += w * (mu_k - mu_T) ** 2
 
    return float(sigma_B2 / sigma_T2)