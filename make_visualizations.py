from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from src.moga import ALGORITHMS, PROBLEMS, load_reference_front

DIMS = [10, 30]


def parse_args():
    p = argparse.ArgumentParser(description="Generate metric, front, convergence, and replay visualizations.")
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--hyperparam-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--make-metric-plots", action="store_true")
    p.add_argument("--make-front-plots", action="store_true")
    p.add_argument("--make-convergence-plots", action="store_true")
    p.add_argument("--make-animations", action="store_true")
    p.add_argument("--make-hyperparam-plots", action="store_true")
    p.add_argument("--representative-run", choices=["median_igd", "best_igd", "run1"], default="median_igd")
    p.add_argument("--fps", type=int, default=2)
    return p.parse_args()


def ensure_dirs(base: Path) -> dict[str, Path]:
    dirs = {
        "metrics": base / "metrics",
        "fronts": base / "fronts",
        "convergence": base / "convergence",
        "animations": base / "animations",
        "hyperparams": base / "hyperparams",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def load_summary(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


def load_raw(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "raw_runs.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


def select_run(raw: pd.DataFrame, algorithm: str, problem: str, dim: int, mode: str) -> int:
    g = raw[(raw.algorithm == algorithm) & (raw.problem == problem) & (raw.dim == dim)].copy()
    if len(g) == 0:
        return 1
    if mode == "run1":
        return int(g.sort_values("run").iloc[0]["run"])
    if mode == "best_igd":
        return int(g.sort_values("igd").iloc[0]["run"])
    med = g["igd"].median()
    g["dist"] = (g["igd"] - med).abs()
    return int(g.sort_values("dist").iloc[0]["run"])


def make_metric_plots(summary: pd.DataFrame, out: Path):
    algorithms = [a for a in ALGORITHMS if a in set(summary.algorithm)]
    for problem in sorted(summary.problem.unique()):
        for metric in ["gd", "igd", "spacing", "hypervolume"]:
            fig, ax = plt.subplots(figsize=(10, 5))
            width = 0.35
            x = np.arange(len(algorithms))
            vals10 = []
            vals30 = []
            for alg in algorithms:
                h10 = summary[(summary.algorithm == alg) & (summary.problem == problem) & (summary.dim == 10)]
                h30 = summary[(summary.algorithm == alg) & (summary.problem == problem) & (summary.dim == 30)]
                vals10.append(float(h10[f"{metric}_mean"].iloc[0]) if len(h10) else np.nan)
                vals30.append(float(h30[f"{metric}_mean"].iloc[0]) if len(h30) else np.nan)
            ax.bar(x - width / 2, vals10, width, label="D=10")
            ax.bar(x + width / 2, vals30, width, label="D=30")
            ax.set_xticks(x)
            ax.set_xticklabels(algorithms, rotation=30, ha="right")
            ax.set_title(f"{problem}: {metric.upper()} comparison")
            ax.set_ylabel(metric.upper())
            if metric in {"gd", "igd", "spacing"}:
                ax.set_yscale("log")
                ax.set_ylabel(f"{metric.upper()} (log scale, lower is better)")
            else:
                ax.set_ylabel(f"{metric.upper()} (higher is better)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / f"{problem}_{metric}_algorithms.png", dpi=200)
            plt.close(fig)

    for metric in ["gd", "igd"]:
        pivot = summary.pivot_table(index=["problem", "dim"], columns="algorithm", values=f"{metric}_mean")
        mat = np.log10(pivot.replace(0, np.nan).to_numpy(dtype=float))
        fig, ax = plt.subplots(figsize=(11, 6))
        im = ax.imshow(mat, aspect="auto")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f"{p} D={d}" for p, d in pivot.index])
        ax.set_title(f"log10 mean {metric.upper()} heatmap")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(f"log10({metric.upper()})")
        fig.tight_layout()
        fig.savefig(out / f"heatmap_{metric}_log10.png", dpi=200)
        plt.close(fig)


def make_front_plots(results_dir: Path, raw: pd.DataFrame, out: Path, representative_mode: str):
    algorithms = [a for a in ALGORITHMS if a in set(raw.algorithm)]
    for problem in sorted(raw.problem.unique()):
        PF = load_reference_front(problem)
        for dim in sorted(raw.dim.unique()):
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(PF[:, 0], PF[:, 1], s=10, alpha=0.35, label="True PF")
            for alg in algorithms:
                run = select_run(raw, alg, problem, int(dim), representative_mode)
                fpath = results_dir / "fronts" / f"{alg}_{problem}_D{int(dim)}_run{run}_front.csv"
                if not fpath.exists():
                    continue
                F = pd.read_csv(fpath)[["f1", "f2"]].to_numpy()
                ax.scatter(F[:, 0], F[:, 1], s=14, alpha=0.75, label=f"{alg} run {run}")
            ax.set_title(f"{problem} D={int(dim)} final nondominated fronts ({representative_mode})")
            ax.set_xlabel("f1")
            ax.set_ylabel("f2")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out / f"{problem}_D{int(dim)}_final_fronts.png", dpi=220)
            plt.close(fig)


def make_convergence_plots(results_dir: Path, out: Path):
    hpath = results_dir / "history" / "history_metrics.csv"
    if not hpath.exists():
        print(f"No history metrics found at {hpath}; skipping convergence plots.")
        return
    hist = pd.read_csv(hpath)
    for problem in sorted(hist.problem.unique()):
        for dim in sorted(hist.dim.unique()):
            for metric in ["gd", "igd", "spacing", "hypervolume"]:
                fig, ax = plt.subplots(figsize=(9, 5))
                for alg, g in hist[(hist.problem == problem) & (hist.dim == dim)].groupby("algorithm"):
                    s = g.groupby("evals", as_index=False)[metric].mean().sort_values("evals")
                    ax.plot(s["evals"], s[metric], marker="o", markersize=3, label=alg)
                ax.set_title(f"{problem} D={int(dim)} {metric.upper()} convergence")
                ax.set_xlabel("Function evaluations")
                ax.set_ylabel(metric.upper())
                if metric in {"gd", "igd", "spacing"}:
                    ax.set_yscale("log")
                    ax.set_ylabel(f"{metric.upper()} (log scale, lower is better)")
                else:
                    ax.set_ylabel(f"{metric.upper()} (higher is better)")
                ax.legend(fontsize=8)
                fig.tight_layout()
                fig.savefig(out / f"{problem}_D{int(dim)}_{metric}_convergence.png", dpi=200)
                plt.close(fig)


def make_animations(results_dir: Path, raw: pd.DataFrame, out: Path, representative_mode: str, fps: int):
    hpath = results_dir / "history" / "history_fronts.csv"
    if not hpath.exists():
        print(f"No history fronts found at {hpath}; skipping animations.")
        return
    hist = pd.read_csv(hpath)
    algorithms = [a for a in ALGORITHMS if a in set(hist.algorithm)]
    for problem in sorted(hist.problem.unique()):
        PF = load_reference_front(problem)
        for dim in sorted(hist.dim.unique()):
            selected_runs = {alg: select_run(raw, alg, problem, int(dim), representative_mode) for alg in algorithms}
            sub = hist[(hist.problem == problem) & (hist.dim == dim)].copy()
            checkpoints = sorted(sub.evals.unique())
            if not checkpoints:
                continue
            ncols = len(algorithms)
            fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4), squeeze=False)
            axes = axes[0]
            all_f1 = list(PF[:, 0])
            all_f2 = list(PF[:, 1])
            for alg in algorithms:
                g = sub[(sub.algorithm == alg) & (sub.run == selected_runs[alg])]
                all_f1.extend(g.f1.tolist())
                all_f2.extend(g.f2.tolist())
            xlim = (min(all_f1), max(all_f1))
            ylim = (min(all_f2), max(all_f2))
            xpad = 0.05 * max(1e-9, xlim[1] - xlim[0])
            ypad = 0.05 * max(1e-9, ylim[1] - ylim[0])

            def update(frame_idx):
                ev = checkpoints[frame_idx]
                for ax, alg in zip(axes, algorithms):
                    ax.clear()
                    ax.scatter(PF[:, 0], PF[:, 1], s=8, alpha=0.25, label="True PF")
                    g = sub[(sub.algorithm == alg) & (sub.run == selected_runs[alg]) & (sub.evals == ev)]
                    if len(g):
                        ax.scatter(g.f1, g.f2, s=16, alpha=0.75, label="Current front")
                    ax.set_title(f"{alg}\nevals={int(ev)}")
                    ax.set_xlim(xlim[0] - xpad, xlim[1] + xpad)
                    ax.set_ylim(ylim[0] - ypad, ylim[1] + ypad)
                    ax.set_xlabel("f1")
                    ax.set_ylabel("f2")
                fig.suptitle(f"{problem} D={int(dim)} Pareto-front replay ({representative_mode})")
                fig.tight_layout()
                return axes

            anim = FuncAnimation(fig, update, frames=len(checkpoints), interval=1000 / max(1, fps), blit=False)
            gif_path = out / f"{problem}_D{int(dim)}_replay.gif"
            try:
                anim.save(gif_path, writer=PillowWriter(fps=max(1, fps)))
            except Exception as exc:
                print(f"Could not save GIF {gif_path}: {exc}")
            plt.close(fig)


def make_hyperparam_plots(hyperparam_dir: Path, out: Path):
    case_path = hyperparam_dir / "hyperparam_case_summary.csv"
    rank_path = hyperparam_dir / "hyperparam_profile_ranking.csv"
    if not case_path.exists() or not rank_path.exists():
        print("Hyperparameter summary files missing; skipping hyperparameter plots.")
        return
    case = pd.read_csv(case_path)
    rank = pd.read_csv(rank_path)
    for alg, g in rank.groupby("algorithm"):
        fig, ax = plt.subplots(figsize=(10, 5))
        g = g.sort_values("mean_igd_across_cases")
        ax.bar(g["tag"], g["mean_igd_across_cases"])
        ax.set_yscale("log")
        ax.set_title(f"{alg}: profile sensitivity by mean IGD")
        ax.set_ylabel("Mean IGD across cases (log scale, lower is better)")
        ax.set_xlabel("Profile")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(out / f"{alg}_profile_igd_sensitivity.png", dpi=200)
        plt.close(fig)

    # Epsilon-specific plot for epsilon mode.
    eps = case[(case.algorithm == "moga_crowding_epsilon") & (case.tag.str.contains("epsilon|default", regex=True))]
    if len(eps):
        s = eps.groupby("tag", as_index=False).agg(igd=("igd_mean", "mean"), gd=("gd_mean", "mean"), front_size=("front_size_mean", "mean"))
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(s["tag"], s["igd"])
        ax.set_yscale("log")
        ax.set_title("moga_crowding_epsilon: epsilon sensitivity")
        ax.set_ylabel("Mean IGD across cases (log scale)")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(out / "moga_crowding_epsilon_epsilon_sensitivity.png", dpi=200)
        plt.close(fig)


def main():
    args = parse_args()
    out_base = args.out_dir if args.out_dir is not None else args.results_dir / "plots"
    dirs = ensure_dirs(out_base)
    summary = load_summary(args.results_dir)
    raw = load_raw(args.results_dir)
    if args.make_metric_plots or not any([args.make_front_plots, args.make_convergence_plots, args.make_animations, args.make_hyperparam_plots]):
        make_metric_plots(summary, dirs["metrics"])
    if args.make_front_plots:
        make_front_plots(args.results_dir, raw, dirs["fronts"], args.representative_run)
    if args.make_convergence_plots:
        make_convergence_plots(args.results_dir, dirs["convergence"])
    if args.make_animations:
        make_animations(args.results_dir, raw, dirs["animations"], args.representative_run, args.fps)
    if args.make_hyperparam_plots and args.hyperparam_dir is not None:
        make_hyperparam_plots(args.hyperparam_dir, dirs["hyperparams"])
    print(f"Plots written under {out_base}")


if __name__ == "__main__":
    main()
