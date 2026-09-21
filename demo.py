import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "Objective_functions"))

import image_utils
from otsu import otsu_fitness, precompute_cumulative
from Kapur_implementation import kapur_entropy
from tsallis import tsallis

# 1. Load image and build PDF
name, gray, hist, pdf = next(image_utils.iter_dataset_images("./BDS500"))
P, S = precompute_cumulative(pdf)

# 2. Test threshold vector for K=3
sample_thresholds = np.array([3, 5, 7, 9, 11, 12])

# 3. Evaluate objective functions
otsu_score = otsu_fitness(sample_thresholds, P, S)
kapur_score = kapur_entropy(sample_thresholds, pdf)
tsallis_score = tsallis(sample_thresholds, hist, q=0.8)

print(f"Image: {name}")
print(f"Otsu Variance (f_Otsu)  : {otsu_score:.4f}")
print(f"Kapur Entropy (f_Kapur) : {kapur_score:.4f}")
print(f"Tsallis Entropy (q=0.8) : {tsallis_score:.4f}")