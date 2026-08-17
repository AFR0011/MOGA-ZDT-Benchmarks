from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.moga import ALGORITHMS, PROBLEMS


def parse_args():
    p = argparse.ArgumentParser(description="Generate a comprehensive Markdown report from MOGA benchmark results.")
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--hyperparam-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("../final_report"))
    return p.parse_args()


def fmt(x):
    if pd.isna(x):
        return ""
    x = float(x)
    if abs(x) < 1e-4 and x != 0:
        return f"{x:.4e}"
    return f"{x:.6g}"


def table_for(summary: pd.DataFrame, metric: str, dim: int, algorithms: list[str]) -> list[str]:
    lines = []
    lines.append("| Problem | " + " | ".join(algorithms) + " |")
    lines.append("|---|" + "---:|" * len(algorithms))
    for problem in PROBLEMS:
        row = []
        for alg in algorithms:
            hit = summary[(summary.algorithm == alg) & (summary.problem == problem) & (summary.dim == dim)]
            row.append(fmt(hit[f"{metric}_mean"].iloc[0]) if len(hit) else "")
        lines.append("| " + problem + " | " + " | ".join(row) + " |")
    return lines


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.results_dir / "summary.csv")
    raw = pd.read_csv(args.results_dir / "raw_runs.csv")
    algorithms = [a for a in ALGORITHMS if a in set(summary.algorithm)]

    lines: list[str] = []
    lines.append("# Comprehensive MOGA ZDT Benchmark Study")
    lines.append("")
    lines.append("## 1. Assignment Objective")
    lines.append("")
    lines.append("The study evaluates GD and IGD for ZDT1, ZDT2, ZDT3, ZDT4, and ZDT6 at D=10 and D=30 using at most 200,000 function evaluations. Baseline MOGA results are reported separately from the extended variants.")
    lines.append("")
    lines.append("## 2. Algorithms Compared")
    lines.append("")
    lines.append("| Mode | What it does | Main purpose |")
    lines.append("|---|---|---|")
    lines.append("| `moga` | Binary-coded baseline MOGA, roulette-style selection, uniform crossover, bit mutation | Assignment baseline |")
    lines.append("| `moga_bonus` | Tournament selection, elitism, adaptive mutation, external nondominated archive | Better convergence and preservation |")
    lines.append("| `moga_crowding` | Adds crowding distance to selection/elitism/archive pruning | Better diversity and front coverage |")
    lines.append("| `moga_crowding_epsilon` | Adds epsilon-grid archive filtering on top of crowding | Removes near-duplicates and controls archive spread |")
    lines.append("| `moga_crowding_hv` | Adds hypervolume-contribution archive pruning on top of crowding | Preserves points contributing most to dominated objective-space area |")
    lines.append("")
    lines.append("## 3. Required Results: MOGA Only")
    lines.append("")
    for dim in [10, 30]:
        for metric in ["gd", "igd"]:
            lines.append(f"### {metric.upper()} for D={dim}")
            lines.extend(table_for(summary, metric, dim, ["moga"]))
            lines.append("")
    lines.append("## 4. Extended Bonus Results: All Modes")
    lines.append("")
    for dim in [10, 30]:
        for metric in ["gd", "igd"]:
            lines.append(f"### {metric.upper()} for D={dim}")
            lines.extend(table_for(summary, metric, dim, algorithms))
            lines.append("")
    lines.append("## 5. Overall Ranking by Mean IGD")
    lines.append("")
    rank = summary.groupby("algorithm", as_index=False).agg(mean_igd=("igd_mean", "mean"), median_igd=("igd_mean", "median"), mean_gd=("gd_mean", "mean"), mean_front_size=("front_size_mean", "mean"))
    rank = rank.sort_values("mean_igd")
    lines.append("| Rank | Algorithm | Mean IGD | Median IGD | Mean GD | Mean front size |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for i, r in enumerate(rank.itertuples(index=False), start=1):
        lines.append(f"| {i} | {r.algorithm} | {fmt(r.mean_igd)} | {fmt(r.median_igd)} | {fmt(r.mean_gd)} | {fmt(r.mean_front_size)} |")
    lines.append("")
    lines.append("## 6. Hyperparameter Sensitivity")
    lines.append("")
    if args.hyperparam_dir and (args.hyperparam_dir / "hyperparam_profile_ranking.csv").exists():
        hp = pd.read_csv(args.hyperparam_dir / "hyperparam_profile_ranking.csv")
        lines.append("The hyperparameter grid was intentionally compact and profile-based. It screens whether mutation strength, archive size, tournament pressure, crossover probability, and epsilon resolution materially affect performance without turning the experiment into a compute bonfire.")
        lines.append("")
        lines.append("| Algorithm | Best profile by mean IGD | Mean IGD | Mean GD |")
        lines.append("|---|---|---:|---:|")
        for alg, g in hp.groupby("algorithm"):
            b = g.sort_values("mean_igd_across_cases").iloc[0]
            lines.append(f"| {alg} | {b['tag']} | {fmt(b['mean_igd_across_cases'])} | {fmt(b['mean_gd_across_cases'])} |")
    else:
        lines.append("No hyperparameter directory was provided, or the hyperparameter ranking file was not found.")
    lines.append("")
    lines.append("## 7. Visualization Guide")
    lines.append("")
    lines.append("- `plots/metrics/`: bar charts and heatmaps comparing GD, IGD, spacing, and hypervolume.")
    lines.append("- `plots/fronts/`: true Pareto front overlaid with representative final fronts from each algorithm.")
    lines.append("- `plots/convergence/`: GD/IGD/spacing/hypervolume over evaluations, if history was saved.")
    lines.append("- `plots/animations/`: GIF replays of how solutions move toward or away from the Pareto front, if history fronts were saved.")
    lines.append("- `plots/hyperparams/`: profile sensitivity plots for the grid search.")
    lines.append("")
    lines.append("## 8. Interpretation Rules")
    lines.append("")
    lines.append("Lower GD and IGD are better. GD measures closeness from obtained solutions to the reference Pareto front. IGD measures how well the reference front is covered by the obtained solution set. Since archive-based algorithms often return many more nondominated points than the baseline, IGD and visual front coverage should be emphasized more than GD alone.")
    lines.append("")
    lines.append("Hypervolume is reported as an extra quality indicator. It is useful, but it is sensitive to the reference point. The hypervolume-contribution archive may improve quality by preserving points with large exclusive dominated area, but it is not guaranteed to win every GD/IGD case.")
    lines.append("")
    lines.append("## 9. Reproducibility")
    lines.append("")
    lines.append("All raw run data is stored in `raw_runs.csv`. Summary statistics are stored in `summary.csv`. The commands used to produce the full study are listed in the root-level `commands.txt` file.")

    out = args.out_dir / "comprehensive_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
