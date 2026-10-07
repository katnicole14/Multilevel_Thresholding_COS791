"""
compare_repair.py

Compares SHADE / L-SHADE before the fix (decimal, unsorted population that is
only repaired inside the objective) with the fixed version (population
repaired and stored as integer thresholds). Same initial populations, same
FE budget; Wilcoxon signed-rank test per configuration.

The old version is read from git commit d2dd871, so no old file is kept
in the repo.

Usage (from the repo root, images in ./BDS500):
    python scripts/compare_repair.py
"""
import importlib.util, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from functools import partial

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OLD_COMMIT = "d2dd871"
old_source = subprocess.run(
    ["git", "show", f"{OLD_COMMIT}:algorithms/Shade.py"],
    cwd=REPO_ROOT, check=True, capture_output=True, text=True,
).stdout
old_path = Path(tempfile.mkdtemp()) / "old_shade.py"
old_path.write_text(old_source)
spec = importlib.util.spec_from_file_location("old_shade", old_path)
old_shade = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_shade)
from Histogram import Process
from Objective_functions.otsu import otsu_fitness, precompute_cumulative
from Objective_functions.Kapur import kapur_entropy
from Objective_functions.tsallis import tsallis
from algorithms.threshold_repair import repair_thresholds
from algorithms.Shade import run_shade, run_lshade
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  # old version produces NaN warnings

POP, FES, SEEDS = 30, 3000, range(15)
rows = []
for img in ["img1", "img4", "img7"]:
    hist, pdf = Process.build_histogram(Process.load_grayscale(REPO_ROOT / "BDS500" / f"{img}.png"))
    P, S = precompute_cumulative(pdf)
    objs = {"Otsu": partial(otsu_fitness, P=P, S=S),
            "Tsallis": lambda t, h=hist: tsallis(t, h, 0.8),
            # old code crashes on Kapur, so give it a version that repairs first
            "Kapur": lambda t, p=pdf: kapur_entropy(repair_thresholds(t), p)}
    for K in (5, 12):
        for oname, f in objs.items():
            for alg in ("SHADE", "L-SHADE"):
                old, new = [], []
                for s in SEEDS:
                    pop = np.random.default_rng(s).uniform(1, 254, (POP, K))
                    if alg == "SHADE":
                        bo = old_shade.shade(K, f, pop_size=POP, max_generations=FES // POP - 1, seed=s)
                        bn = run_shade(pop, f, s, FES)
                    else:
                        bo = old_shade.l_shade(K, f, max_nfes=FES, pop_size_init=POP, seed=s)
                        bn = run_lshade(pop, f, s, FES)
                    # score the old result on its REPAIRED thresholds (what you'd actually use)
                    old.append(f(repair_thresholds(bo[0]))); new.append(bn["best_fitness"])
                old, new = np.array(old), np.array(new)
                try: p = wilcoxon(new, old).pvalue
                except ValueError: p = 1.0
                win = "new" if p < 0.05 and new.mean() > old.mean() else "old" if p < 0.05 else "tie"
                rows.append((img, K, oname, alg, old.mean(), old.std(), new.mean(), new.std(), p, win))
                print(f"{img} K={K:2d} {oname:7s} {alg:7s} old {old.mean():10.3f}±{old.std():7.3f}  new {new.mean():10.3f}±{new.std():7.3f}  p={p:.3f} -> {win}", flush=True)
import warnings
from collections import Counter
print(Counter(r[-1] for r in rows))
for K in (5,12): print("K", K, Counter(r[-1] for r in rows if r[1]==K))
