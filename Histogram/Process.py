#Installing dependies
from pathlib import Path
from typing import Iterator, Tuple

import matplotlib.pyplot as plt
import math

import numpy as np
from PIL import Image
"""
DATA PREPERATION
1. loading the dataset(Images)
2. Converting the images to greyscale
3. Doing a histogram
"""
#Method for loading a grey scale

def load_grayscale(path: Path) -> np.ndarray:
    """
    Load an image file and convert it to 8-bit grayscale.
    """
    img = Image.open(path).convert("RGB")   # drop alpha if present
    img = img.convert("L")                  # RGB -> grayscale, uint8
    return np.array(img, dtype=np.uint8)


def crop_white_border(gray: np.ndarray, value: int = 255) -> np.ndarray:
    """
    Remove outer rows/columns that are entirely `value` (pure white).

    The BDS500 copies in this project have a 5-13 px white frame that is not
    part of the photo. Left in, it adds an artificial histogram spike at 255
    (7-11% of all pixels) that the objectives waste a threshold on and that
    distorts PSNR/SSIM/U. Only full rows/columns at the edges are removed.
    """
    rows = np.where(~np.all(gray == value, axis=1))[0]
    cols = np.where(~np.all(gray == value, axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return gray  # completely white image: nothing sensible to crop
    return gray[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

#METHOD FOR IMPLEMENTING THE HISTOGRAM BASED ON THE FORMULA GIVEM
def build_histogram(gray: np.ndarray, levels: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the raw histogram h(i) and the normalised probability
    distribution p_i = h(i) / (M*N) used by Otsu/Kapur/Tsallis.
    """
    hist, _ = np.histogram(gray.ravel(), bins=levels, range=(0, levels))
    pdf = hist.astype(np.float64) / gray.size
    return hist, pdf

def is_ground_truth_file(path: Path) -> bool:
    """BDS500 ground-truth / ground-boundary files are suffixed '_gt'."""
    return path.stem.endswith("_gt")



def iter_dataset_images(
    dataset_dir: str,
    pattern: str = "*.png",
) -> Iterator[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generator that yields ONE image at a time from a dataset folder,
    skipping any ground-truth ('*_gt.png') files.

    Yields
    ------
    name : str            filename stem, e.g. 'img1'
    gray : np.ndarray     (H, W) uint8 grayscale image
    hist : np.ndarray     (256,) raw histogram counts
    pdf  : np.ndarray     (256,) normalised probability distribution
    """
    folder = Path(dataset_dir)
    files = sorted(
        p for p in folder.glob(pattern) if not is_ground_truth_file(p)
    )

    for path in files:
        gray = load_grayscale(path)
        hist, pdf = build_histogram(gray)
        yield path.stem, gray, hist, pdf


