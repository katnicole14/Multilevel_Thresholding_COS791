"""
run_experiment1.py

Experiment 1 (BSD500): every DE variant x every objective x every K x 30
independent runs on every image, with the same function-evaluation budget.
Each run's best thresholds are used to segment the image, and PSNR, SSIM
and Uniformity are computed against the original.

One JSON line per run is appended to <out>/runs.jsonl, so an interrupted
experiment can simply be started again and continues where it stopped.
Run settings are saved to <out>/config.json (use it for the report's
parameter table).

Usage (from the repo root, images in ./BDS500):
    python experiments/run_experiment1.py                 # full experiment
    python experiments/run_experiment1.py --quick         # 2-minute smoke test
    python experiments/run_experiment1.py --images img1 img2 --k 3 12 --runs 5

Then build the tables and figures:
    python experiments/tables_experiment.py --experiment 1
    python experiments/figures_experiment.py --experiment 1
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache, partial
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from Histogram import Process
from Objective_functions.otsu import otsu_fitness, precompute_cumulative
from Objective_functions.Kapur import kapur_entropy
from Objective_functions.tsallis import tsallis
from algorithms.jade import run_jade
from algorithms.late_acceptance import run_lade
from algorithms.Shade import run_shade, run_lshade
from algorithms.SelectMainLP import run_de
from evaluation.metrics import psnr, ssim, uniformity
from evaluation.segmentation import segment

K_LEVELS = [3, 5, 7, 9, 11, 12]
OBJECTIVES = ["Otsu", "Kapur", "Tsallis"]
ALGORITHMS = ["DE", "JADE", "SHADE", "L-SHADE", "LADE"]
TSALLIS_Q = 0.8
POP_SIZE = 30                 # NP for DE, JADE, SHADE and LADE
LSHADE_NP_PER_DIMENSION = 18  # L-SHADE: NP_init = 18 * D
CONVERGENCE_POINTS = 50       # best-so-far fitness is stored at this many FE checkpoints

RUNNERS = {
    "DE": partial(run_de, scale_factor=0.5, crossover_rate=0.9),
    "JADE": partial(run_jade, p_best_rate=0.1, learning_rate=0.1),
    "SHADE": partial(run_shade, p=0.1),
    "L-SHADE": partial(run_lshade, pop_size_min=4, memory_size=6, p=0.11, arch_rate=2.6),
    "LADE": partial(run_lade, scale_factor=0.5, crossover_rate=0.9, history_length=10),
}

PARAMETERS = {
    "DE": "DE/rand/1/bin, NP=30, F=0.5, CR=0.9",
    "JADE": "current-to-pbest/1, NP=30, p=0.1, c=0.1, mu_F=mu_CR=0.5 initially, archive size NP",
    "SHADE": "current-to-pbest/1, NP=30, H=NP, p=0.1, M_F=M_CR=0.5 initially, archive size NP",
    "L-SHADE": "current-to-pbest/1, NP_init=18K, NP_min=4, H=6, p=0.11, archive rate 2.6, LPSR",
    "LADE": "DE/rand/1/bin, NP=30, F=0.5, CR=0.9, late-acceptance history length L=10",
}


@lru_cache(maxsize=None)
def load_image(image_path):
    """Grey image with the white frame removed, plus its histogram (cached per worker)."""
    gray = Process.crop_white_border(Process.load_grayscale(Path(image_path)))
    hist, pdf = Process.build_histogram(gray)
    return gray, hist, pdf


def build_objective(name, hist, pdf):
    if name == "Otsu":
        P, S = precompute_cumulative(pdf)
        return partial(otsu_fitness, P=P, S=S)
    if name == "Kapur":
        return partial(kapur_entropy, probabilities=pdf)
    if name == "Tsallis":
        return lambda t: tsallis(t, hist, TSALLIS_Q)
    raise ValueError(f"Unknown objective {name}")


def initial_population(k, run, size):
    """
    Same random start for every algorithm and objective in a given (K, run).
    L-SHADE asks for more rows; its first POP_SIZE rows are the same as the
    other algorithms' population because rows are drawn in order.
    """
    rng = np.random.default_rng([run, k])
    return rng.uniform(1, 254, size=(size, k))


def sample_convergence(history, max_fes):
    """Best-so-far fitness at CONVERGENCE_POINTS evenly spaced FE checkpoints."""
    fes = np.array([h[0] for h in history])
    best = np.array([h[1] for h in history], dtype=float)
    checkpoints = np.linspace(max_fes / CONVERGENCE_POINTS, max_fes, CONVERGENCE_POINTS)
    index = np.searchsorted(fes, checkpoints, side="right") - 1
    values = np.where(index >= 0, best[np.clip(index, 0, None)], np.nan)
    return [None if np.isnan(v) else round(float(v), 6) for v in values]


def run_one(task):
    """Run one (image, objective, K, algorithm, run) combination. Executed in a worker."""
    gray, hist, pdf = load_image(task["image_path"])
    objective = build_objective(task["objective"], hist, pdf)
    k, run, max_fes = task["k"], task["run"], task["max_fes"]

    size = LSHADE_NP_PER_DIMENSION * k if task["algorithm"] == "L-SHADE" else POP_SIZE
    population = initial_population(k, run, size)

    start = time.perf_counter()
    result = RUNNERS[task["algorithm"]](population, objective, run, max_fes)
    elapsed = time.perf_counter() - start

    thresholds = result["best_thresholds"]
    segmented = segment(gray, thresholds)
    return {
        "dataset": "BSD500",
        "image": task["image"],
        "objective": task["objective"],
        "K": k,
        "algorithm": task["algorithm"],
        "run": run,
        "seed": run,
        "max_fes": max_fes,
        "fes_used": result["function_evaluations"],
        "best_fitness": result["best_fitness"],
        "thresholds": thresholds,
        "time_s": round(elapsed, 4),
        "psnr": psnr(gray, segmented),
        "ssim": ssim(gray, segmented),
        "uniformity": uniformity(gray, thresholds),
        "convergence": sample_convergence(result["convergence_history"], max_fes),
    }


def task_key(record):
    return (record["image"], record["objective"], record["K"], record["algorithm"], record["run"])


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: DE variants on BSD500.")
    parser.add_argument("--images-dir", default="BDS500")
    parser.add_argument("--images", nargs="*", help="image names (default: all non-_gt PNGs)")
    parser.add_argument("--k", type=int, nargs="*", default=K_LEVELS)
    parser.add_argument("--objectives", nargs="*", default=OBJECTIVES, choices=OBJECTIVES)
    parser.add_argument("--algorithms", nargs="*", default=ALGORITHMS, choices=ALGORITHMS)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--fes", type=int, default=5000, help="max function evaluations per run")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--out", default="results/experiment1")
    parser.add_argument("--quick", action="store_true",
                        help="smoke test: 2 images, K=3 and 12, 2 runs, 1000 FEs")
    args = parser.parse_args()

    if args.quick:
        args.images = args.images or ["img1", "img2"]
        args.k, args.runs, args.fes = [3, 12], 2, 1000
        args.out = args.out + "_quick"

    images_dir = REPO_ROOT / args.images_dir
    if args.images:
        image_paths = [images_dir / f"{name}.png" for name in args.images]
    else:
        image_paths = sorted(p for p in images_dir.glob("*.png") if not Process.is_ground_truth_file(p))
    missing = [p for p in image_paths if not p.exists()]
    if not image_paths or missing:
        sys.exit(f"Images not found in {images_dir}: {missing or 'no PNG files'}")

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_file = out_dir / "runs.jsonl"

    config = {
        "dataset": "BSD500",
        "images": [p.stem for p in image_paths],
        "K": args.k,
        "objectives": args.objectives,
        "algorithms": args.algorithms,
        "runs": args.runs,
        "max_function_evaluations": args.fes,
        "tsallis_q": TSALLIS_Q,
        "preprocessing": "greyscale (ITU-R 601-2 luma), white border cropped",
        "segmentation": "pixels replaced by the mean grey level of their class",
        "initial_population": "uniform in [1, 254], same for all algorithms per (K, run)",
        "parameters": {a: PARAMETERS[a] for a in args.algorithms},
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    done = set()
    if runs_file.exists():
        with runs_file.open() as f:
            done = {task_key(json.loads(line)) for line in f if line.strip()}

    tasks = [
        {"image": p.stem, "image_path": str(p), "objective": objective, "k": k,
         "algorithm": algorithm, "run": run, "max_fes": args.fes}
        for k in args.k
        for p in image_paths
        for objective in args.objectives
        for run in range(args.runs)
        for algorithm in args.algorithms
    ]
    todo = [t for t in tasks if (t["image"], t["objective"], t["k"], t["algorithm"], t["run"]) not in done]
    print(f"{len(tasks)} runs in total, {len(tasks) - len(todo)} already done, {len(todo)} to go "
          f"({args.workers} workers, {args.fes} FEs each). Writing to {runs_file.relative_to(REPO_ROOT)}",
          flush=True)
    if not todo:
        return

    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool, runs_file.open("a") as out:
        futures = [pool.submit(run_one, task) for task in todo]
        for finished, future in enumerate(as_completed(futures), start=1):
            out.write(json.dumps(future.result()) + "\n")
            if finished % 100 == 0 or finished == len(todo):
                out.flush()
                elapsed = time.time() - start
                eta = elapsed / finished * (len(todo) - finished)
                print(f"  {finished}/{len(todo)} runs  elapsed {elapsed / 60:.1f} min  "
                      f"ETA {eta / 60:.1f} min", flush=True)

    print(f"Done in {(time.time() - start) / 60:.1f} min.")


if __name__ == "__main__":
    main()
