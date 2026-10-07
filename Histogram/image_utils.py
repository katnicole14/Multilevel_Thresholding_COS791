from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
from PIL import Image


#Method for loading a grey scale

def load_grayscale(path: Path) -> np.ndarray:
    """
    Load an image file and convert it to 8-bit grayscale.

    Uses PIL's 'L' mode conversion, which applies the standard
    ITU-R 601-2 luma transform:
        L = 0.299*R + 0.587*G + 0.114*B
    Any alpha channel is dropped automatically by first forcing RGB.
    """
    img = Image.open(path).convert("RGB")   # drop alpha if present
    img = img.convert("L")                  # RGB -> grayscale, uint8
    return np.array(img, dtype=np.uint8)

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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Quick sanity check over the BDS500 folder
    for name, gray, hist, pdf in iter_dataset_images("/content/sample_data/BSDS500"):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(gray, cmap="gray", vmin=0,vmax=255)
        axes[0].set_title(name+"GreyScale")
        axes[0].axis("off")

        axes[1].plot(hist)
        axes[1].set_title(name +"Histogram")
        axes[1].set_xlabel("Intensity level")
        axes[1].set_ylabel("Pixel Count")

        plt.tight_layout()
        plt.show()
        print(f"{name:10s} shape={gray.shape}  sum(pdf)={pdf.sum():.4f}")
