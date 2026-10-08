"""
tables_experiment1.py

Builds the Experiment 1 tables from <results>/runs.jsonl:

  summary_<metric>.{csv,md,tex}    mean ± std over all images and runs,
                                   rows = objective x algorithm, columns = K
  per_image_<metric>.csv           mean ± std per image (appendix detail)
  ranks_<metric>.csv               average rank of each algorithm per objective
                                   and K (rank 1 = best, computed per image/run)

Metrics: PSNR, SSIM, Uniformity (higher is better), best fitness (higher is
better) and run time in seconds (lower is better). In the Markdown and LaTeX
tables the best algorithm per objective and K is in bold.

Usage:
    python experiments/tables_experiment1.py
    python experiments/tables_experiment1.py --results results/experiment1_quick
"""

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

METRICS = {
    # column: (title, decimals, higher_is_better)
    "psnr": ("PSNR (dB)", 2, True),
    "ssim": ("SSIM", 4, True),
    "uniformity": ("Uniformity U", 4, True),
    "best_fitness": ("Best objective value", 4, True),
    "time_s": ("Run time (s)", 3, False),
}
ALGORITHM_ORDER = ["DE", "JADE", "SHADE", "L-SHADE", "LADE"]
OBJECTIVE_ORDER = ["Otsu", "Kapur", "Tsallis"]


def load_runs(results_dir):
    with (results_dir / "runs.jsonl").open() as f:
        runs = pd.DataFrame(json.loads(line) for line in f if line.strip())
    runs["algorithm"] = pd.Categorical(runs["algorithm"], [a for a in ALGORITHM_ORDER if a in set(runs["algorithm"])])
    runs["objective"] = pd.Categorical(runs["objective"], [o for o in OBJECTIVE_ORDER if o in set(runs["objective"])])
    return runs


def summary(runs, metric):
    """Mean and std per (objective, algorithm, K) over all images and runs."""
    grouped = runs.groupby(["objective", "algorithm", "K"], observed=True)[metric]
    return grouped.agg(["mean", "std"]).reset_index()


def best_per_block(stats, higher_is_better, decimals):
    """
    Set of (objective, algorithm, K) with the best mean for each (objective, K).
    Algorithms that tie at the displayed precision are all marked best.
    """
    best = set()
    for (objective, k), block in stats.groupby(["objective", "K"], observed=True):
        shown = block["mean"].round(decimals)
        target = shown.max() if higher_is_better else shown.min()
        for algorithm in block.loc[shown == target, "algorithm"]:
            best.add((objective, algorithm, k))
    return best


def to_wide(stats, decimals, best, bold):
    """Rows = (objective, algorithm), columns = K, cells = 'mean ± std'."""
    def cell(row):
        text = f"{row['mean']:.{decimals}f} ± {row['std']:.{decimals}f}"
        if (row["objective"], row["algorithm"], row["K"]) in best:
            text = bold(text)
        return text

    stats = stats.assign(cell=stats.apply(cell, axis=1))
    wide = stats.pivot(index=["objective", "algorithm"], columns="K", values="cell")
    wide.columns = [f"K={k}" for k in wide.columns]
    return wide


def write_markdown(wide, title, path):
    lines = [f"### {title}", "", "| Objective | Algorithm | " + " | ".join(wide.columns) + " |",
             "|---|---|" + "---|" * len(wide.columns)]
    previous = None
    for (objective, algorithm), row in wide.iterrows():
        shown = objective if objective != previous else ""
        previous = objective
        lines.append(f"| {shown} | {algorithm} | " + " | ".join(row.values) + " |")
    path.write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def write_latex(wide, title, label, path):
    columns = "ll" + "c" * len(wide.columns)
    lines = [
        "\\begin{table*}[t]", "\\centering", "\\scriptsize",
        f"\\caption{{{title} (mean $\\pm$ std over all images and runs; best per objective and $K$ in bold).}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{columns}}}", "\\toprule",
        "Objective & Algorithm & " + " & ".join(f"$K={c[2:]}$" for c in wide.columns) + " \\\\",
        "\\midrule",
    ]
    previous = None
    for (objective, algorithm), row in wide.iterrows():
        if previous is not None and objective != previous:
            lines.append("\\midrule")
        shown = objective if objective != previous else ""
        previous = objective
        cells = [c.replace("±", "$\\pm$") for c in row.values]
        lines.append(f"{shown} & {algorithm} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    path.write_text("\n".join(lines) + "\n")


def per_image(runs, metric, decimals):
    grouped = runs.groupby(["image", "objective", "algorithm", "K"], observed=True)[metric]
    stats = grouped.agg(["mean", "std"]).reset_index()
    stats["cell"] = stats.apply(lambda r: f"{r['mean']:.{decimals}f} ± {r['std']:.{decimals}f}", axis=1)
    wide = stats.pivot(index=["image", "objective", "algorithm"], columns="K", values="cell")
    wide.columns = [f"K={k}" for k in wide.columns]
    return wide


def average_ranks(runs, metric, higher_is_better):
    """Rank the algorithms within every (image, objective, K, run), then average."""
    ranked = runs.copy()
    ranked["rank"] = ranked.groupby(["image", "objective", "K", "run"], observed=True)[metric].rank(
        ascending=not higher_is_better, method="average")
    table = ranked.groupby(["objective", "K", "algorithm"], observed=True)["rank"].mean().unstack("algorithm")
    return table.round(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/experiment1")
    args = parser.parse_args()

    results_dir = REPO_ROOT / args.results
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(results_dir)
    print(f"{len(runs)} runs: {runs['image'].nunique()} images, K={sorted(runs['K'].unique())}, "
          f"{runs['run'].nunique()} runs each\n")

    for metric, (title, decimals, higher_is_better) in METRICS.items():
        stats = summary(runs, metric)
        best = best_per_block(stats, higher_is_better, decimals)
        stats.to_csv(tables_dir / f"summary_{metric}.csv", index=False)

        markdown = write_markdown(to_wide(stats, decimals, best, lambda s: f"**{s}**"),
                                  title, tables_dir / f"summary_{metric}.md")
        write_latex(to_wide(stats, decimals, best, lambda s: f"\\textbf{{{s}}}"),
                    f"{title} on BSD500", f"tab:exp1_{metric}", tables_dir / f"summary_{metric}.tex")
        per_image(runs, metric, decimals).to_csv(tables_dir / f"per_image_{metric}.csv")
        average_ranks(runs, metric, higher_is_better).to_csv(tables_dir / f"ranks_{metric}.csv")

        if metric != "best_fitness":
            print(markdown, "\n")

    print(f"Tables written to {tables_dir.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
