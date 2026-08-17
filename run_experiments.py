from __future__ import annotations

# Keep each worker single-threaded internally. With 30 processes, letting BLAS/OpenMP
# also spawn threads is how laptops become small space heaters.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.moga import ALGORITHMS, GivenMOGAConfig, PROBLEMS, run_given_moga

DIMS = [10, 30]


def parse_args():
    parser = argparse.ArgumentParser(description="Run MOGA benchmark experiments.")
    parser.add_argument("--problems", nargs="+", default=PROBLEMS, choices=PROBLEMS)
    parser.add_argument("--dims", nargs="+", default=DIMS, type=int, choices=DIMS)
    parser.add_argument("--algorithms", nargs="+", default=["moga"], choices=ALGORITHMS)
    parser.add_argument("--max-evals", type=int, default=200_000)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=538)
    parser.add_argument("--pop-size", type=int, default=None, help="Override population size. Default preserves given code: 10*D.")
    parser.add_argument("--bits-per-var", type=int, default=30)
    parser.add_argument("--p-crossover", type=float, default=0.9)
    parser.add_argument("--p-mutation", type=float, default=0.1)
    parser.add_argument("--tournament-size", type=int, default=7)
    parser.add_argument("--elite-rate", type=float, default=0.10)
    parser.add_argument("--archive-size", type=int, default=1000)
    parser.add_argument("--epsilon", type=float, default=0.005)
    parser.add_argument("--config-file", type=Path, default=None, help="Optional JSON mapping algorithm -> parameter overrides.")
    parser.add_argument("--tag", default="default", help="Experiment tag stored in CSV outputs.")
    parser.add_argument("--out-dir", type=Path, default=Path("../results"))
    parser.add_argument("--save-fronts", action="store_true", default=False)
    parser.add_argument("--save-history", action="store_true", default=False)
    parser.add_argument("--history-interval", type=int, default=10_000)
    parser.add_argument("--history-max-points", type=int, default=250)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes. Use 30 for your requested parallel run.")
    return parser.parse_args()


def load_config_overrides(config_file: Path | None) -> dict[str, dict[str, Any]]:
    if config_file is None:
        return {}
    with open(config_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Accept either {"moga": {...}} or {"algorithms": {"moga": {...}}}
    if "algorithms" in data and isinstance(data["algorithms"], dict):
        return data["algorithms"]
    return data


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["tag", "algorithm", "problem", "dim"]
    rows = []
    for keys, g in raw.groupby(group_cols, sort=False):
        tag, alg, problem, dim = keys
        rows.append({
            "tag": tag,
            "algorithm": alg,
            "problem": problem,
            "dim": dim,
            "gd_mean": g["gd"].mean(),
            "gd_std": g["gd"].std(ddof=1) if len(g) > 1 else 0.0,
            "gd_min": g["gd"].min(),
            "gd_max": g["gd"].max(),
            "igd_mean": g["igd"].mean(),
            "igd_std": g["igd"].std(ddof=1) if len(g) > 1 else 0.0,
            "igd_min": g["igd"].min(),
            "igd_max": g["igd"].max(),
            "spacing_mean": g["spacing"].mean(),
            "spacing_std": g["spacing"].std(ddof=1) if len(g) > 1 else 0.0,
            "hypervolume_mean": g["hypervolume"].mean(),
            "hypervolume_std": g["hypervolume"].std(ddof=1) if len(g) > 1 else 0.0,
            "actual_evals_mean": g["actual_evals"].mean(),
            "front_size_mean": g["front_size"].mean(),
            "runtime_sec_mean": g["runtime_sec"].mean(),
        })
    return pd.DataFrame(rows)


def write_required_tables(summary: pd.DataFrame, out_dir: Path, algorithms: list[str], problems: list[str], dims: list[int]):
    tables_dir = out_dir / "tables"
    required_dir = out_dir / "required_tables"
    bonus_dir = out_dir / "bonus_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    required_dir.mkdir(parents=True, exist_ok=True)
    bonus_dir.mkdir(parents=True, exist_ok=True)

    # Assignment-required MOGA-only tables.
    for dim in dims:
        for metric in ["gd", "igd"]:
            rows = []
            for problem in problems:
                hit = summary[(summary.algorithm == "moga") & (summary.problem == problem) & (summary.dim == dim)]
                rows.append({"Problems": problem, "MOGA": float(hit[f"{metric}_mean"].iloc[0]) if len(hit) else None})
            pd.DataFrame(rows).to_csv(required_dir / f"{metric}_D{dim}_MOGA_only.csv", index=False)

    # Extended all-algorithm tables.
    for dim in dims:
        for metric in ["gd", "igd", "spacing", "hypervolume"]:
            rows = []
            for problem in problems:
                row = {"Problems": problem}
                for alg in algorithms:
                    hit = summary[(summary.algorithm == alg) & (summary.problem == problem) & (summary.dim == dim)]
                    row[alg] = float(hit[f"{metric}_mean"].iloc[0]) if len(hit) else None
                rows.append(row)
            pd.DataFrame(rows).to_csv(tables_dir / f"{metric}_D{dim}.csv", index=False)
            pd.DataFrame(rows).to_csv(bonus_dir / f"{metric}_D{dim}_all_algorithms.csv", index=False)

    # Relative improvement tables vs baseline for minimization metrics.
    for metric in ["gd", "igd", "spacing"]:
        rows = []
        for dim in dims:
            for problem in problems:
                base = summary[(summary.algorithm == "moga") & (summary.problem == problem) & (summary.dim == dim)]
                if not len(base):
                    continue
                base_val = float(base[f"{metric}_mean"].iloc[0])
                row = {"problem": problem, "dim": dim, "baseline_moga": base_val}
                for alg in algorithms:
                    hit = summary[(summary.algorithm == alg) & (summary.problem == problem) & (summary.dim == dim)]
                    row[alg] = base_val / float(hit[f"{metric}_mean"].iloc[0]) if len(hit) and float(hit[f"{metric}_mean"].iloc[0]) > 0 else None
                rows.append(row)
        pd.DataFrame(rows).to_csv(bonus_dir / f"relative_{metric}_improvement_vs_moga.csv", index=False)


def write_report(summary: pd.DataFrame, out_dir: Path, algorithms: list[str], problems: list[str], dims: list[int], workers: int, args: argparse.Namespace):
    lines = []
    lines.append("# MOGA ZDT Benchmark Report")
    lines.append("")
    lines.append("## Experiment Setup")
    lines.append("")
    lines.append("- Problems: ZDT1, ZDT2, ZDT3, ZDT4, ZDT6")
    lines.append("- Dimensions: D=10 and D=30")
    lines.append(f"- Maximum function evaluations: {args.max_evals}")
    lines.append(f"- Independent runs per case: {args.n_runs}")
    lines.append("- Baseline: binary-coded MOGA adapted from the provided code")
    lines.append("- Required metrics: GD and IGD using the same formulas as the provided MATLAB functions")
    lines.append("- Extra analysis metrics: spacing and 2D hypervolume")
    lines.append(f"- Parallel execution workers used by the runner: {workers}")
    lines.append("")
    lines.append("## Required Assignment Tables: MOGA Only")
    lines.append("")
    lines.append("Baseline tables are written separately from extended algorithm comparisons so the original MOGA results remain distinguishable from the improved variants.")
    lines.append("")
    for dim in dims:
        for metric in ["gd", "igd"]:
            lines.append(f"### Required {metric.upper()} Results for D={dim}")
            lines.append("")
            lines.append("| Problem | MOGA |")
            lines.append("|---|---:|")
            for problem in problems:
                hit = summary[(summary.algorithm == "moga") & (summary.problem == problem) & (summary.dim == dim)]
                val = f"{float(hit[f'{metric}_mean'].iloc[0]):.10g}" if len(hit) else ""
                lines.append(f"| {problem} | {val} |")
            lines.append("")
    lines.append("## Bonus Algorithms")
    lines.append("")
    lines.append("- `moga`: given-code-style baseline using binary chromosomes, roulette selection, uniform crossover, and bit mutation.")
    lines.append("- `moga_bonus`: tournament selection, elitism, adaptive mutation, and an external nondominated archive pruned by uniform spread along f1.")
    lines.append("- `moga_crowding`: the bonus method plus NSGA-II-style crowding distance in selection, elitism, and archive pruning.")
    lines.append("- `moga_crowding_epsilon`: crowding-distance MOGA plus epsilon-grid archive control, keeping one representative per objective-space cell before pruning.")
    lines.append("- `moga_crowding_hv`: crowding-distance MOGA plus hypervolume-contribution archive pruning, repeatedly removing the nondominated point with the smallest exclusive hypervolume contribution.")
    lines.append("")
    lines.append("## Extended Bonus Comparison")
    lines.append("")
    for dim in dims:
        for metric in ["gd", "igd"]:
            lines.append(f"### {metric.upper()} Results for D={dim}: All Algorithms")
            lines.append("")
            header = "| Problem | " + " | ".join(algorithms) + " |"
            sep = "|---|" + "---:|" * len(algorithms)
            lines.append(header)
            lines.append(sep)
            for problem in problems:
                vals = []
                for alg in algorithms:
                    hit = summary[(summary.algorithm == alg) & (summary.problem == problem) & (summary.dim == dim)]
                    vals.append(f"{float(hit[f'{metric}_mean'].iloc[0]):.10g}" if len(hit) else "")
                lines.append("| " + problem + " | " + " | ".join(vals) + " |")
            lines.append("")
    lines.append("## Interpretation Notes")
    lines.append("")
    lines.append("Lower GD and IGD are better. GD measures closeness of obtained solutions to the true Pareto front. IGD measures how well the true reference front is covered by the obtained solution set. Since archive-based algorithms may return many more nondominated points than the baseline, GD can look artificially strong; therefore IGD and final-front visual coverage should be emphasized in the final discussion.")
    lines.append("")
    lines.append("The hypervolume-contribution mode is the fifth and most indicator-driven archive method. It is expected to improve diversity and preserve points that contribute most to the dominated objective-space area, but it may not always beat crowding or epsilon on IGD because hypervolume is sensitive to the reference point and can favor certain regions of the front.")
    lines.append("")
    lines.append("## Evaluation-Budget Note")
    lines.append("")
    lines.append("The code never exceeds the requested maximum function-evaluation budget. With the default population size of 10*D, D=10 usually reaches 200000 exactly, while D=30 may stop at 199800 evaluations to avoid crossing 200000.")
    lines.append("")
    (out_dir / "final_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_jobs(args: argparse.Namespace, config_overrides: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    jobs = []
    total = len(args.algorithms) * len(args.problems) * len(args.dims) * args.n_runs
    job_index = 0
    for alg in args.algorithms:
        alg_overrides = config_overrides.get(alg, {})
        for dim in args.dims:
            for problem in args.problems:
                for run in range(1, args.n_runs + 1):
                    job_index += 1
                    seed = args.seed + 10000 * run + 100 * dim + PROBLEMS.index(problem)
                    jobs.append({
                        "job_index": job_index,
                        "total": total,
                        "tag": args.tag,
                        "algorithm": alg,
                        "problem": problem,
                        "dim": dim,
                        "run": run,
                        "seed": seed,
                        "max_evals": args.max_evals,
                        "pop_size": args.pop_size,
                        "bits_per_var": args.bits_per_var,
                        "p_crossover": alg_overrides.get("p_crossover", args.p_crossover),
                        "p_mutation": alg_overrides.get("p_mutation", args.p_mutation),
                        "tournament_size": alg_overrides.get("tournament_size", args.tournament_size),
                        "elite_rate": alg_overrides.get("elite_rate", args.elite_rate),
                        "archive_size": alg_overrides.get("archive_size", args.archive_size),
                        "epsilon": alg_overrides.get("epsilon", args.epsilon),
                        "history_interval": args.history_interval if args.save_history else None,
                        "history_max_points": args.history_max_points,
                        "verbose": args.verbose,
                    })
    return jobs


def execute_job(job: dict[str, Any]) -> tuple[dict[str, Any], Any, list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = GivenMOGAConfig(
        problem=job["problem"],
        dim=job["dim"],
        max_evals=job["max_evals"],
        pop_size=job["pop_size"],
        bits_per_var=job["bits_per_var"],
        p_crossover=job["p_crossover"],
        p_mutation=job["p_mutation"],
        seed=job["seed"],
        algorithm=job["algorithm"],
        tournament_size=job["tournament_size"],
        elite_rate=job["elite_rate"],
        archive_size=job["archive_size"],
        epsilon=job["epsilon"],
        history_interval=job["history_interval"],
        history_max_points=job["history_max_points"],
        verbose=job["verbose"],
        tag=job["tag"],
    )
    result = run_given_moga(cfg)
    # Import here to avoid a wider public API just for metrics.
    from src.moga import spacing_metric, hypervolume_2d
    row = {
        "job_index": job["job_index"],
        "tag": job["tag"],
        "algorithm": job["algorithm"],
        "problem": job["problem"],
        "dim": job["dim"],
        "run": job["run"],
        "seed": job["seed"],
        "max_evals": job["max_evals"],
        "actual_evals": result.actual_evals,
        "pop_size": result.config.pop_size or 10 * job["dim"],
        "bits_per_var": result.config.bits_per_var,
        "p_crossover": result.config.p_crossover,
        "p_mutation": result.config.p_mutation,
        "tournament_size": result.config.tournament_size,
        "elite_rate": result.config.elite_rate,
        "archive_size": result.config.archive_size,
        "epsilon": result.config.epsilon,
        "front_size": result.front_size,
        "gd": result.gd,
        "igd": result.igd,
        "spacing": spacing_metric(result.F),
        "hypervolume": hypervolume_2d(result.F),
        "runtime_sec": result.runtime_sec,
    }
    # Attach job identifiers to history rows.
    hm = []
    for r in result.history_metrics:
        rr = {"tag": job["tag"], "algorithm": job["algorithm"], "problem": job["problem"], "dim": job["dim"], "run": job["run"], "seed": job["seed"], **r}
        hm.append(rr)
    hf = []
    for r in result.history_fronts:
        rr = {"tag": job["tag"], "algorithm": job["algorithm"], "problem": job["problem"], "dim": job["dim"], "run": job["run"], "seed": job["seed"], **r}
        hf.append(rr)
    return row, result.F, hm, hf


def save_partial(raw_rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not raw_rows:
        return
    pd.DataFrame(raw_rows).sort_values("job_index").to_csv(out_dir / "raw_runs.csv", index=False)


def main():
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_fronts:
        (out_dir / "fronts").mkdir(exist_ok=True)
    if args.save_history:
        (out_dir / "history").mkdir(exist_ok=True)

    config_overrides = load_config_overrides(args.config_file)
    jobs = build_jobs(args, config_overrides)
    total = len(jobs)
    workers = max(1, int(args.workers))
    rows: list[dict[str, Any]] = []
    history_metrics_rows: list[dict[str, Any]] = []
    history_front_rows: list[dict[str, Any]] = []

    print(f"Running {total} jobs with {workers} worker(s).")
    if workers == 1:
        for job in jobs:
            print(f"[{job['job_index']}/{total}] {job['algorithm']} {job['problem']} D={job['dim']} run={job['run']} seed={job['seed']}")
            row, front, hm, hf = execute_job(job)
            rows.append(row)
            history_metrics_rows.extend(hm)
            history_front_rows.extend(hf)
            if args.save_fronts:
                pd.DataFrame(front, columns=["f1", "f2"]).to_csv(out_dir / "fronts" / f"{row['algorithm']}_{row['problem']}_D{row['dim']}_run{row['run']}_front.csv", index=False)
            save_partial(rows, out_dir)
    else:
        print("Parallel mode: output order follows job completion, not job submission. Because time, regrettably, has no manners.")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_job = {executor.submit(execute_job, job): job for job in jobs}
            completed = 0
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                completed += 1
                try:
                    row, front, hm, hf = future.result()
                except Exception as exc:
                    print(f"FAILED [{job['job_index']}/{total}] {job['algorithm']} {job['problem']} D={job['dim']} run={job['run']} seed={job['seed']}: {exc}")
                    raise
                rows.append(row)
                history_metrics_rows.extend(hm)
                history_front_rows.extend(hf)
                print(f"[{completed}/{total}] done job={row['job_index']} {row['algorithm']} {row['problem']} D={row['dim']} run={row['run']} GD={row['gd']:.6g} IGD={row['igd']:.6g}")
                if args.save_fronts:
                    pd.DataFrame(front, columns=["f1", "f2"]).to_csv(out_dir / "fronts" / f"{row['algorithm']}_{row['problem']}_D{row['dim']}_run{row['run']}_front.csv", index=False)
                save_partial(rows, out_dir)

    raw = pd.DataFrame(rows).sort_values("job_index").drop(columns=["job_index"])
    raw.to_csv(out_dir / "raw_runs.csv", index=False)
    summary = summarize(raw)
    summary.to_csv(out_dir / "summary.csv", index=False)
    write_required_tables(summary, out_dir, args.algorithms, args.problems, args.dims)
    write_report(summary, out_dir, args.algorithms, args.problems, args.dims, workers, args)

    if args.save_history and history_metrics_rows:
        pd.DataFrame(history_metrics_rows).to_csv(out_dir / "history" / "history_metrics.csv", index=False)
    if args.save_history and history_front_rows:
        pd.DataFrame(history_front_rows).to_csv(out_dir / "history" / "history_fronts.csv", index=False)

    print(f"\nDone. Results written to {out_dir}")


if __name__ == "__main__":
    main()
