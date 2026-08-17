from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

import pandas as pd

from run_experiments import execute_job
from src.moga import ALGORITHMS, PROBLEMS

DIMS = [10, 30]

# Deliberately small, profile-based grid. This is not a full factorial furnace.
GRID_PROFILES: dict[str, dict[str, Any]] = {
    "default": {
        "p_crossover": 0.90,
        "p_mutation": 0.10,
        "tournament_size": 7,
        "elite_rate": 0.10,
        "archive_size": 1000,
        "epsilon": 0.005,
    },
    "mutation_low": {
        "p_crossover": 0.90,
        "p_mutation": 0.05,
        "tournament_size": 7,
        "elite_rate": 0.10,
        "archive_size": 1000,
        "epsilon": 0.005,
    },
    "mutation_high": {
        "p_crossover": 0.90,
        "p_mutation": 0.20,
        "tournament_size": 7,
        "elite_rate": 0.10,
        "archive_size": 1000,
        "epsilon": 0.005,
    },
    "crossover_low": {
        "p_crossover": 0.70,
        "p_mutation": 0.10,
        "tournament_size": 7,
        "elite_rate": 0.10,
        "archive_size": 1000,
        "epsilon": 0.005,
    },
    "tournament_small": {
        "p_crossover": 0.90,
        "p_mutation": 0.10,
        "tournament_size": 3,
        "elite_rate": 0.10,
        "archive_size": 1000,
        "epsilon": 0.005,
    },
    "tournament_large": {
        "p_crossover": 0.90,
        "p_mutation": 0.10,
        "tournament_size": 11,
        "elite_rate": 0.10,
        "archive_size": 1000,
        "epsilon": 0.005,
    },
    "archive_small": {
        "p_crossover": 0.90,
        "p_mutation": 0.10,
        "tournament_size": 7,
        "elite_rate": 0.10,
        "archive_size": 250,
        "epsilon": 0.005,
    },
    "archive_large": {
        "p_crossover": 0.90,
        "p_mutation": 0.10,
        "tournament_size": 7,
        "elite_rate": 0.10,
        "archive_size": 2000,
        "epsilon": 0.005,
    },
    "epsilon_fine": {
        "p_crossover": 0.90,
        "p_mutation": 0.10,
        "tournament_size": 7,
        "elite_rate": 0.10,
        "archive_size": 1000,
        "epsilon": 0.0025,
    },
    "epsilon_coarse": {
        "p_crossover": 0.90,
        "p_mutation": 0.10,
        "tournament_size": 7,
        "elite_rate": 0.10,
        "archive_size": 1000,
        "epsilon": 0.010,
    },
}

# Smaller default selection for faster screening. Include all profiles by passing --profiles all.
PRESET_TINY = [
    "default",
    "mutation_low",
    "mutation_high",
    "archive_small",
    "archive_large",
    "epsilon_fine",
    "epsilon_coarse",
]


def parse_args():
    p = argparse.ArgumentParser(description="Hyperparameter sensitivity/grid screening for MOGA variants.")
    p.add_argument("--problems", nargs="+", default=PROBLEMS, choices=PROBLEMS)
    p.add_argument("--dims", nargs="+", default=DIMS, type=int, choices=DIMS)
    p.add_argument("--algorithms", nargs="+", default=ALGORITHMS, choices=ALGORITHMS)
    p.add_argument("--profiles", nargs="+", default=PRESET_TINY, help="Profile names, or 'all'.")
    p.add_argument("--max-evals", type=int, default=50_000)
    p.add_argument("--n-runs", type=int, default=3)
    p.add_argument("--seed", type=int, default=9538)
    p.add_argument("--pop-size", type=int, default=None)
    p.add_argument("--bits-per-var", type=int, default=30)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out-dir", type=Path, default=Path("../results_hyperparam_tiny"))
    return p.parse_args()


def build_jobs(args) -> list[dict[str, Any]]:
    profiles = list(GRID_PROFILES.keys()) if args.profiles == ["all"] else args.profiles
    for name in profiles:
        if name not in GRID_PROFILES:
            raise ValueError(f"Unknown profile {name}. Available: {list(GRID_PROFILES)}")

    jobs = []
    total = len(profiles) * len(args.algorithms) * len(args.problems) * len(args.dims) * args.n_runs
    job_index = 0
    for profile in profiles:
        params = GRID_PROFILES[profile]
        for alg in args.algorithms:
            for dim in args.dims:
                for problem in args.problems:
                    for run in range(1, args.n_runs + 1):
                        job_index += 1
                        seed = args.seed + 100000 * list(GRID_PROFILES.keys()).index(profile) + 10000 * run + 100 * dim + PROBLEMS.index(problem)
                        jobs.append({
                            "job_index": job_index,
                            "total": total,
                            "tag": profile,
                            "algorithm": alg,
                            "problem": problem,
                            "dim": dim,
                            "run": run,
                            "seed": seed,
                            "max_evals": args.max_evals,
                            "pop_size": args.pop_size,
                            "bits_per_var": args.bits_per_var,
                            "p_crossover": params["p_crossover"],
                            "p_mutation": params["p_mutation"],
                            "tournament_size": params["tournament_size"],
                            "elite_rate": params["elite_rate"],
                            "archive_size": params["archive_size"],
                            "epsilon": params["epsilon"],
                            "history_interval": None,
                            "history_max_points": 0,
                            "verbose": False,
                        })
    return jobs


def summarize(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_case = raw.groupby(["tag", "algorithm", "problem", "dim"], as_index=False).agg(
        gd_mean=("gd", "mean"),
        gd_std=("gd", "std"),
        igd_mean=("igd", "mean"),
        igd_std=("igd", "std"),
        spacing_mean=("spacing", "mean"),
        hypervolume_mean=("hypervolume", "mean"),
        front_size_mean=("front_size", "mean"),
        runtime_sec_mean=("runtime_sec", "mean"),
    )
    profile_rank = per_case.groupby(["algorithm", "tag"], as_index=False).agg(
        mean_igd_across_cases=("igd_mean", "mean"),
        median_igd_across_cases=("igd_mean", "median"),
        mean_gd_across_cases=("gd_mean", "mean"),
        mean_spacing_across_cases=("spacing_mean", "mean"),
        mean_runtime_sec=("runtime_sec_mean", "mean"),
    ).sort_values(["algorithm", "mean_igd_across_cases"])
    return per_case, profile_rank


def write_best_configs(profile_rank: pd.DataFrame, out_dir: Path) -> None:
    best: dict[str, dict[str, Any]] = {"algorithms": {}}
    for alg, g in profile_rank.groupby("algorithm", sort=False):
        best_profile = str(g.sort_values("mean_igd_across_cases").iloc[0]["tag"])
        best["algorithms"][alg] = GRID_PROFILES[best_profile] | {"selected_profile": best_profile}
    with open(out_dir / "best_configs.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)


def main():
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args)
    rows = []
    workers = max(1, args.workers)
    print(f"Running {len(jobs)} hyperparameter jobs with {workers} worker(s).")
    if workers == 1:
        for job in jobs:
            print(f"[{job['job_index']}/{job['total']}] {job['tag']} {job['algorithm']} {job['problem']} D={job['dim']} run={job['run']}")
            row, _, _, _ = execute_job(job)
            rows.append(row)
            pd.DataFrame(rows).sort_values("job_index").to_csv(out_dir / "hyperparam_raw.csv", index=False)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_job = {executor.submit(execute_job, job): job for job in jobs}
            completed = 0
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                completed += 1
                row, _, _, _ = future.result()
                rows.append(row)
                print(f"[{completed}/{len(jobs)}] {row['tag']} {row['algorithm']} {row['problem']} D={row['dim']} run={row['run']} IGD={row['igd']:.6g}")
                pd.DataFrame(rows).sort_values("job_index").to_csv(out_dir / "hyperparam_raw.csv", index=False)

    raw = pd.DataFrame(rows).sort_values("job_index").drop(columns=["job_index"])
    raw.to_csv(out_dir / "hyperparam_raw.csv", index=False)
    per_case, profile_rank = summarize(raw)
    per_case.to_csv(out_dir / "hyperparam_case_summary.csv", index=False)
    profile_rank.to_csv(out_dir / "hyperparam_profile_ranking.csv", index=False)
    write_best_configs(profile_rank, out_dir)
    with open(out_dir / "grid_profiles.json", "w", encoding="utf-8") as f:
        json.dump({k: GRID_PROFILES[k] for k in (list(GRID_PROFILES.keys()) if args.profiles == ["all"] else args.profiles)}, f, indent=2)
    print(f"Done. Hyperparameter results written to {out_dir}")


if __name__ == "__main__":
    main()
