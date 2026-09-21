import Process.py
import Otsu.py

from skimage.filters import threshold_multiotsu, threshold_otsu
from pathlib import Path # Add missing import for Path

if __name__ == '__main__':
    # Quick sanity check over the BDS500 folder

    p= Process()
    O=otsu()
    
    for name, gray, hist, pdf in p.iter_dataset_images("/content/BSDS500"):
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

    #Checking if the Ostu implementation works
    # Use pathlib.Path for consistency with load_grayscale type hint

    gray = p.load_grayscale(Path("/content/BSDS500/img1.png"))
    hist, pdf = p.build_histogram(gray)
    P, S = O.precompute_cumulative(pdf)

    # --- K=1 (classic binary Otsu) ---
    ref_t = threshold_otsu(gray)
    print(f"[K=1] skimage threshold_otsu = {ref_t}")

    best_val, best_t = -1, None
    for cand in range(1, 255):
        val = O.otsu_fitness(np.array([cand]), P, S)
        if val > best_val:
            best_val, best_t = val, cand
    print(f"[K=1] brute-force otsu_fitness best threshold = {best_t}  (value={best_val:.6f})")

    # --- K=2 (compare against skimage multi-Otsu) ---
    ref_ts = O.threshold_multiotsu(gray, classes=3)
    print(f"\n[K=2] skimage threshold_multiotsu = {list(ref_ts)}")

    val_at_ref = O.otsu_fitness(np.array(ref_ts), P, S)
    print(f"[K=2] otsu_fitness at skimage's thresholds = {val_at_ref:.6f}")

    # small local brute-force search around skimage's answer to confirm it's (near) optimal
    best_val2, best_pair = -1, None
    for a in range(1, 255):
        for b in range(a + 1, 255):
            val = O.otsu_fitness(np.array([a, b]), P, S)
            if val > best_val2:
                best_val2, best_pair = val, (a, b)
    print(f"[K=2] brute-force best pair = {best_pair}  (value={best_val2:.6f})")