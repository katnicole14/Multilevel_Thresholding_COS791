# Multilevel_Thresholding_COS791
Project For Image Analysis and Understanding 

Multilevel image thresholding with five Differential Evolution variants
(DE, JADE, SHADE, L-SHADE, Late Acceptance DE) and three objective functions
(Otsu, Kapur, Tsallis).

## Setup

```bash
pip install -r requirements.txt
```

Images are not in git. Put the BSD500 images (`img1.png` ... `img10.png`)
in `BDS500/` at the repo root. Run every command from the repo root.

## Layout

| Folder | Contents |
|---|---|
| `algorithms/` | `run_de`, `run_jade`, `run_shade`, `run_lshade`, `run_lade`, shared mutation / crossover / threshold repair |
| `Objective_functions/` | Otsu, Kapur, Tsallis |
| `Histogram/` | image loading, white-border crop, histogram |
| `evaluation/` | segmentation, PSNR, SSIM, Uniformity, class separability |
| `experiments/` | master scripts that reproduce the results |
| `scripts/` | sanity checks and demos |
| `tests/` | pytest suite |

All algorithms share one interface:

```python
result = run_X(initial_population, objective_function, seed, maximum_function_evaluations)
result["best_thresholds"], result["best_fitness"], result["convergence_history"]
```

## Experiment 1 (BSD500)

```bash
python experiments/run_experiment1.py --quick              # smoke test, under a minute
python experiments/run_experiment1.py                      # full: 10 images x 3 objectives x 6 K x 5 algorithms x 30 runs
python experiments/tables_experiment.py --experiment 1     # mean ± std tables (CSV, Markdown, LaTeX)
python experiments/figures_experiment.py --experiment 1    # segmentation, convergence and time-vs-K figures
python experiments/stats_tests.py --experiment 1           # Friedman + Wilcoxon (Holm) tests
```

## Experiment 2 (CHAOS MRI)

Put the 15 CHAOS MR slices (PNG) in `CHAOS/`. Segmentation quality is
measured with class separability eta = sigma_B^2 / sigma_T^2.

```bash
python experiments/run_experiment2.py --quick
python experiments/run_experiment2.py
python experiments/tables_experiment.py --experiment 2
python experiments/figures_experiment.py --experiment 2
python experiments/stats_tests.py --experiment 2
```

Both experiments append one line per run to `results/experiment<N>/runs.jsonl`
and can be stopped and restarted at any time; finished runs are skipped.
Settings used are saved to `results/experiment<N>/config.json`.

## Checks

```bash
python -m pytest tests
python scripts/check_all.py --k 12     # every algorithm x objective works
python scripts/verify_optimum.py       # all algorithms reach the exhaustive-search optimum (K=2)
python scripts/visual_check.py         # look at one segmentation
python -m scripts.Otsu_test            # Otsu matches scikit-image's multi-Otsu
```
