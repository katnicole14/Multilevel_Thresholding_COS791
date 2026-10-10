"""
stats_tests.py

Friedman and Wilcoxon signed-rank tests on <results>/runs.jsonl.

Friedman: for every objective and K, the five algorithms are ranked within
each (image, run) block (rank 1 = best) and compared with the Friedman test.
An overall test uses every (image, objective, K, run) as a block.

Wilcoxon: every pair of algorithms, for every objective and K, paired by
(image, run). All p-values of one metric are corrected together with the
Holm procedure (alpha = 0.05).

Outputs in <results>/tables/:
  friedman_<metric>.{csv,tex}          chi^2, p and mean rank per objective and K
  friedman_overall.csv                 overall test per metric
  wilcoxon_<metric>.csv                every pairwise test with raw and Holm p-values
  wilcoxon_summary_<metric>.{csv,tex}  significant wins / no difference / losses
                                       of each algorithm against each other one

Usage:
    python experiments/stats_tests.py --experiment 1
    python experiments/stats_tests.py --experiment 2
    python experiments/stats_tests.py --experiment 2 --metrics eta best_fitness
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

REPO_ROOT = Path(__file__).resolve().parent.parent

ALGORITHM_ORDER = ["DE", "JADE", "SHADE", "L-SHADE", "LADE"]
OBJECTIVE_ORDER = ["Otsu", "Kapur", "Tsallis"]
METRICS = {
    # column: (title, higher_is_better)
    "best_fitness": ("Best objective value", True),
    "eta": ("Class separability eta", True),
    "psnr": ("PSNR", True),
    "ssim": ("SSIM", True),
    "uniformity": ("Uniformity U", True),
}
DEFAULT_METRICS = {1: ["best_fitness", "psnr", "ssim", "uniformity"], 2: ["best_fitness", "eta"]}
ALPHA = 0.05


def load_runs(results_dir):
    with (results_dir / "runs.jsonl").open() as f:
        return pd.DataFrame(json.loads(line) for line in f if line.strip())


def holm(p_values):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (len(p) - rank) * p[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted


def blocks(runs, metric, algorithms):
    """Rows = (image, run) blocks, columns = algorithms. Incomplete blocks are dropped."""
    table = runs.pivot_table(index=["image", "run"], columns="algorithm", values=metric)
    return table[algorithms].dropna()


def mean_ranks(table, higher_is_better):
    return table.rank(axis=1, ascending=not higher_is_better, method="average").mean()


def friedman(runs, metric, higher_is_better, algorithms):
    rows = []
    for objective in [o for o in OBJECTIVE_ORDER if o in set(runs["objective"])]:
        for k in sorted(runs["K"].unique()):
            subset = runs[(runs["objective"] == objective) & (runs["K"] == k)]
            table = blocks(subset, metric, algorithms)
            if (table.nunique(axis=1) == 1).all():
                chi2, p = 0.0, 1.0  # every algorithm identical in every block
            else:
                chi2, p = friedmanchisquare(*(table[a] for a in algorithms))
            ranks = mean_ranks(table, higher_is_better)
            rows.append({"objective": objective, "K": k, "blocks": len(table),
                         "chi2": chi2, "p": p, "significant": p < ALPHA,
                         **{a: ranks[a] for a in algorithms}, "best": ranks.idxmin()})
    return pd.DataFrame(rows)


def friedman_overall(runs, metric, higher_is_better, algorithms):
    table = runs.pivot_table(index=["image", "objective", "K", "run"], columns="algorithm",
                             values=metric)[algorithms].dropna()
    chi2, p = friedmanchisquare(*(table[a] for a in algorithms))
    ranks = mean_ranks(table, higher_is_better)
    return {"metric": metric, "blocks": len(table), "chi2": chi2, "p": p,
            "significant": p < ALPHA, **{a: ranks[a] for a in algorithms}, "best": ranks.idxmin()}


def pairwise_wilcoxon(runs, metric, higher_is_better, algorithms):
    rows = []
    for objective in [o for o in OBJECTIVE_ORDER if o in set(runs["objective"])]:
        for k in sorted(runs["K"].unique()):
            subset = runs[(runs["objective"] == objective) & (runs["K"] == k)]
            table = blocks(subset, metric, algorithms)
            for a, b in itertools.combinations(algorithms, 2):
                diff = table[a] - table[b]
                if not higher_is_better:
                    diff = -diff
                if np.allclose(diff, 0):
                    statistic, p = 0.0, 1.0
                else:
                    statistic, p = wilcoxon(table[a], table[b])
                rows.append({"objective": objective, "K": k, "A": a, "B": b, "pairs": len(diff),
                             "A_better": int((diff > 0).sum()), "ties": int(np.isclose(diff, 0).sum()),
                             "B_better": int((diff < 0).sum()), "mean_diff": diff.mean(),
                             "statistic": statistic, "p": p})
    result = pd.DataFrame(rows)
    result["p_holm"] = holm(result["p"])
    result["significant"] = result["p_holm"] < ALPHA
    result["winner"] = np.where(~result["significant"], "none",
                                np.where(result["mean_diff"] > 0, result["A"], result["B"]))
    return result


def wilcoxon_summary(pairs, algorithms):
    """Cell (row X, column Y) = 'wins/no difference/losses' of X against Y over all objectives and K."""
    summary = pd.DataFrame("", index=algorithms, columns=algorithms)
    for x, y in itertools.permutations(algorithms, 2):
        rows = pairs[((pairs["A"] == x) & (pairs["B"] == y)) | ((pairs["A"] == y) & (pairs["B"] == x))]
        wins = int((rows["winner"] == x).sum())
        losses = int((rows["winner"] == y).sum())
        summary.loc[x, y] = f"{wins}/{len(rows) - wins - losses}/{losses}"
    for x in algorithms:
        summary.loc[x, x] = "--"
    return summary


def friedman_latex(table, title, label, algorithms, path):
    lines = [
        "\\begin{table*}[t]", "\\centering", "\\scriptsize",
        f"\\caption{{Friedman test on {title}: mean rank of each algorithm (1 = best), "
        f"$\\chi^2$ and $p$ per objective and $K$; lowest mean rank in bold.}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{ll" + "c" * len(algorithms) + "cc}", "\\toprule",
        "Objective & $K$ & " + " & ".join(algorithms) + " & $\\chi^2$ & $p$ \\\\", "\\midrule",
    ]
    previous = None
    for _, row in table.iterrows():
        if previous is not None and row["objective"] != previous:
            lines.append("\\midrule")
        shown = row["objective"] if row["objective"] != previous else ""
        previous = row["objective"]
        best = min(row[a] for a in algorithms)
        ranks = [f"\\textbf{{{row[a]:.2f}}}" if np.isclose(row[a], best) else f"{row[a]:.2f}"
                 for a in algorithms]
        p = "$<$0.001" if row["p"] < 0.001 else f"{row['p']:.3f}"
        lines.append(f"{shown} & {row['K']} & " + " & ".join(ranks) + f" & {row['chi2']:.2f} & {p} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    path.write_text("\n".join(lines) + "\n")


def summary_latex(summary, title, label, n_tests, path):
    algorithms = list(summary.index)
    lines = [
        "\\begin{table}[t]", "\\centering", "\\scriptsize",
        f"\\caption{{Wilcoxon signed-rank tests on {title} (Holm-corrected, $\\alpha = 0.05$). "
        f"Each cell gives the number of objective/$K$ settings (out of {n_tests}) in which the row "
        f"algorithm is significantly better / not significantly different / significantly worse "
        f"than the column algorithm.}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{l" + "c" * len(algorithms) + "}", "\\toprule",
        " & " + " & ".join(algorithms) + " \\\\", "\\midrule",
        *(f"{x} & " + " & ".join(summary.loc[x]) + " \\\\" for x in algorithms),
        "\\bottomrule", "\\end{tabular}", "\\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, choices=[1, 2], required=True)
    parser.add_argument("--results", help="default: results/experiment<N>")
    parser.add_argument("--metrics", nargs="*", choices=METRICS, help="default: depends on the experiment")
    args = parser.parse_args()

    results_dir = REPO_ROOT / (args.results or f"results/experiment{args.experiment}")
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(results_dir)
    algorithms = [a for a in ALGORITHM_ORDER if a in set(runs["algorithm"])]
    metrics = [m for m in (args.metrics or DEFAULT_METRICS[args.experiment]) if m in runs.columns]

    overall = []
    for metric in metrics:
        title, higher_is_better = METRICS[metric]

        table = friedman(runs, metric, higher_is_better, algorithms)
        table.to_csv(tables_dir / f"friedman_{metric}.csv", index=False)
        friedman_latex(table, title, f"tab:exp{args.experiment}_friedman_{metric}", algorithms,
                       tables_dir / f"friedman_{metric}.tex")
        overall.append(friedman_overall(runs, metric, higher_is_better, algorithms))

        pairs = pairwise_wilcoxon(runs, metric, higher_is_better, algorithms)
        pairs.to_csv(tables_dir / f"wilcoxon_{metric}.csv", index=False)
        summary = wilcoxon_summary(pairs, algorithms)
        summary.to_csv(tables_dir / f"wilcoxon_summary_{metric}.csv")
        n_settings = runs[["objective", "K"]].drop_duplicates().shape[0]
        summary_latex(summary, title, f"tab:exp{args.experiment}_wilcoxon_{metric}", n_settings,
                      tables_dir / f"wilcoxon_summary_{metric}.tex")

        print(f"\n=== {title} ===")
        print(f"Friedman: significant in {int(table['significant'].sum())}/{len(table)} objective/K settings")
        print(f"Wilcoxon: {int(pairs['significant'].sum())}/{len(pairs)} pairwise tests significant after Holm")
        print("Wins/no difference/losses (row vs column):")
        print(summary.to_string())

    overall = pd.DataFrame(overall)
    overall.to_csv(tables_dir / "friedman_overall.csv", index=False)
    print("\n=== Overall Friedman (mean rank, 1 = best) ===")
    print(overall.round(4).to_string(index=False))
    print(f"\nStatistics written to {tables_dir}/")


if __name__ == "__main__":
    main()
