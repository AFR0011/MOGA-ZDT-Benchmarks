from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import numpy as np

PROBLEMS = ["ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6"]
ALGORITHMS = [
    "moga",
    "moga_bonus",
    "moga_crowding",
    "moga_crowding_epsilon",
    "moga_crowding_hv",
]


@dataclass
class GivenMOGAConfig:
    problem: str
    dim: int
    max_evals: int = 200_000
    pop_size: int | None = None
    bits_per_var: int = 30
    p_crossover: float = 0.9
    p_mutation: float = 0.1
    seed: int = 538
    algorithm: str = "moga"
    tournament_size: int = 7
    elite_rate: float = 0.10
    archive_size: int = 1000
    epsilon: float = 0.005
    preserve_extremes: bool = True
    history_interval: int | None = None
    history_max_points: int = 250
    verbose: bool = False
    tag: str = "default"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GivenMOGAResult:
    F: np.ndarray
    gd: float
    igd: float
    actual_evals: int
    generations: int
    front_size: int
    runtime_sec: float
    config: GivenMOGAConfig
    history_metrics: List[Dict[str, Any]] = field(default_factory=list)
    history_fronts: List[Dict[str, Any]] = field(default_factory=list)


def zdt_bounds(problem: str, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    if problem == "ZDT4":
        lower = np.r_[0.0, -5.0 * np.ones(dim - 1)]
        upper = np.r_[1.0, 5.0 * np.ones(dim - 1)]
    else:
        lower = np.zeros(dim)
        upper = np.ones(dim)
    return lower, upper


def evaluate_zdt(X: np.ndarray, problem: str) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, dim = X.shape
    if dim < 2:
        raise ValueError("ZDT problems require dim >= 2")

    if problem == "ZDT1":
        y = X[:, 1:]
        g = 1.0 + 9.0 * y.sum(axis=1) / (dim - 1)
        f1 = X[:, 0]
        f2 = g * (1.0 - np.sqrt(f1 / g))
    elif problem == "ZDT2":
        y = X[:, 1:]
        g = 1.0 + 9.0 * y.sum(axis=1) / (dim - 1)
        f1 = X[:, 0]
        f2 = g * (1.0 - (f1 / g) ** 2)
    elif problem == "ZDT3":
        y = X[:, 1:]
        g = 1.0 + 9.0 * y.sum(axis=1) / (dim - 1)
        f1 = X[:, 0]
        f2 = g * (1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10.0 * np.pi * f1))
    elif problem == "ZDT4":
        y = X[:, 1:]
        g = 1.0 + 10.0 * (dim - 1) + np.sum(y * y - 10.0 * np.cos(4.0 * np.pi * y), axis=1)
        f1 = X[:, 0]
        f2 = g * (1.0 - np.sqrt(f1 / g))
    elif problem == "ZDT6":
        y = X[:, 1:]
        g = 1.0 + 9.0 * (y.sum(axis=1) / (dim - 1)) ** 0.25
        f1 = 1.0 - np.exp(-4.0 * X[:, 0]) * (np.sin(6.0 * np.pi * X[:, 0]) ** 6)
        f2 = g * (1.0 - (f1 / g) ** 2)
    else:
        raise ValueError(f"Unsupported problem: {problem}")
    return np.column_stack([f1, f2])


def load_reference_front(problem: str, package_root: Path | None = None, n_points: int = 2_001) -> np.ndarray:
    """Load a supplied reference front when available, otherwise generate one.

    This keeps the repository standalone. The generated fronts follow the
    analytical Pareto fronts of the ZDT problems used in this study.
    """
    if package_root is None:
        candidates = [
            Path(__file__).resolve().parents[2],
            Path(__file__).resolve().parents[3],
            Path.cwd(),
            Path.cwd().parent,
        ]
    else:
        candidates = [package_root]
    for root in candidates:
        pf = root / "given_code_original" / "MOGA_AU" / "ParetoFront" / f"{problem}.txt"
        if pf.exists():
            return np.loadtxt(pf)

    if problem in {"ZDT1", "ZDT4"}:
        f1 = np.linspace(0.0, 1.0, n_points)
        f2 = 1.0 - np.sqrt(f1)
    elif problem == "ZDT2":
        f1 = np.linspace(0.0, 1.0, n_points)
        f2 = 1.0 - f1**2
    elif problem == "ZDT3":
        segments = [
            (0.0, 0.0830015349),
            (0.1822287280, 0.2577623634),
            (0.4093136748, 0.4538821041),
            (0.6183967944, 0.6525117038),
            (0.8233317983, 0.8518328654),
        ]
        per_segment = max(2, n_points // len(segments))
        f1 = np.concatenate([np.linspace(a, b, per_segment) for a, b in segments])
        f2 = 1.0 - np.sqrt(f1) - f1 * np.sin(10.0 * np.pi * f1)
    elif problem == "ZDT6":
        x = np.linspace(0.0, 1.0, 100_001)
        attainable = 1.0 - np.exp(-4.0 * x) * (np.sin(6.0 * np.pi * x) ** 6)
        f1 = np.linspace(float(attainable.min()), 1.0, n_points)
        f2 = 1.0 - f1**2
    else:
        raise ValueError(f"Unsupported problem: {problem}")
    return np.column_stack([f1, f2])

def decode_bits(bits: np.ndarray, bits_per_var: int, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    n, chrom_len = bits.shape
    dim = chrom_len // bits_per_var
    powers = (2 ** np.arange(bits_per_var - 1, -1, -1, dtype=np.uint64)).astype(np.float64)
    denom = float(2 ** bits_per_var - 1)
    X = np.empty((n, dim), dtype=float)
    for j in range(dim):
        s = j * bits_per_var
        e = s + bits_per_var
        dec = bits[:, s:e].astype(np.float64) @ powers
        X[:, j] = lower[j] + ((upper[j] - lower[j]) * dec) / denom
    return X


def dominance_ranks(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    if len(F) == 0:
        return np.array([], dtype=int)
    dominates = (np.all(F[:, None, :] <= F[None, :, :], axis=2)
                 & np.any(F[:, None, :] < F[None, :, :], axis=2))
    return 1 + dominates.sum(axis=0).astype(int)


def nondominated(F: np.ndarray) -> np.ndarray:
    if len(F) == 0:
        return np.empty((0, 2), dtype=float)
    F = np.unique(np.asarray(F, dtype=float), axis=0)
    ranks = dominance_ranks(F)
    out = F[ranks == 1]
    return out[np.argsort(out[:, 0], kind="mergesort")]


def crowding_distance(F: np.ndarray) -> np.ndarray:
    """NSGA-II crowding distance for one front. Larger is better."""
    F = np.asarray(F, dtype=float)
    n = len(F)
    if n == 0:
        return np.array([], dtype=float)
    if n <= 2:
        return np.full(n, np.inf, dtype=float)
    m = F.shape[1]
    dist = np.zeros(n, dtype=float)
    for j in range(m):
        order = np.argsort(F[:, j], kind="mergesort")
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        f_min = F[order[0], j]
        f_max = F[order[-1], j]
        denom = f_max - f_min
        if denom <= 1e-15:
            continue
        for k in range(1, n - 1):
            if not np.isinf(dist[order[k]]):
                dist[order[k]] += (F[order[k + 1], j] - F[order[k - 1], j]) / denom
    return dist


def population_crowding(F: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    ranks = np.asarray(ranks)
    cd = np.zeros(len(F), dtype=float)
    for r in np.unique(ranks):
        idx = np.flatnonzero(ranks == r)
        cd[idx] = crowding_distance(F[idx])
    return cd


def spacing_metric(F: np.ndarray) -> float:
    """Schott-style spacing metric over neighboring objective distances. Lower is more even."""
    F = nondominated(F)
    if len(F) < 3:
        return 0.0
    d = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=2)
    d[d == 0] = np.inf
    nearest = d.min(axis=1)
    return float(np.sqrt(np.mean((nearest - nearest.mean()) ** 2)))


def prune_archive(F: np.ndarray, max_size: int) -> np.ndarray:
    """Legacy archive pruning: nondominated front + uniform sampling by f1."""
    F = nondominated(F)
    if len(F) <= max_size:
        return F
    order = np.argsort(F[:, 0], kind="mergesort")
    F = F[order]
    idx = np.unique(np.round(np.linspace(0, len(F) - 1, max_size)).astype(int))
    return F[idx]


def prune_archive_crowding(F: np.ndarray, max_size: int) -> np.ndarray:
    """Prune nondominated archive by keeping highest crowding-distance points."""
    F = nondominated(F)
    if len(F) <= max_size:
        return F
    cd = crowding_distance(F)
    keep = np.argsort(-cd, kind="mergesort")[:max_size]
    F = F[keep]
    return F[np.argsort(F[:, 0], kind="mergesort")]


def prune_archive_epsilon(F: np.ndarray, max_size: int, epsilon: float) -> np.ndarray:
    """Epsilon-grid archive: one representative per objective-space cell, then crowding prune."""
    F = nondominated(F)
    if len(F) <= 1:
        return F
    eps = max(float(epsilon), 1e-12)
    mins = F.min(axis=0)
    cells = np.floor((F - mins) / eps).astype(np.int64)
    # Prefer the representative closest to the lower-left corner of each cell.
    best_by_cell: dict[tuple[int, int], tuple[float, int]] = {}
    for i, cell in enumerate(map(tuple, cells)):
        score = float(np.linalg.norm(F[i] - (mins + eps * cells[i])))
        if cell not in best_by_cell or score < best_by_cell[cell][0]:
            best_by_cell[cell] = (score, i)
    keep = [idx for _, idx in best_by_cell.values()]
    F = F[np.asarray(keep, dtype=int)]
    if len(F) > max_size:
        F = prune_archive_crowding(F, max_size)
    return F[np.argsort(F[:, 0], kind="mergesort")]


def _hypervolume_reference(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    maxs = np.nanmax(F, axis=0)
    mins = np.nanmin(F, axis=0)
    span = np.maximum(maxs - mins, 1.0)
    return maxs + 0.10 * span + 1e-9


def hypervolume_2d(F: np.ndarray, ref: np.ndarray | None = None) -> float:
    """Exact 2D hypervolume for minimization, assuming a nondominated front."""
    F = nondominated(F)
    if len(F) == 0:
        return 0.0
    if ref is None:
        ref = _hypervolume_reference(F)
    ref = np.asarray(ref, dtype=float)
    # Keep only points dominated by the reference point.
    F = F[np.all(F < ref, axis=1)]
    if len(F) == 0:
        return 0.0
    F = F[np.argsort(F[:, 0], kind="mergesort")]
    xs = F[:, 0]
    ys = F[:, 1]
    x_next = np.r_[xs[1:], ref[0]]
    widths = np.maximum(0.0, x_next - xs)
    heights = np.maximum(0.0, ref[1] - ys)
    return float(np.sum(widths * heights))


def hypervolume_contributions_2d(F: np.ndarray, ref: np.ndarray | None = None, preserve_extremes: bool = True) -> np.ndarray:
    """Per-point 2D HV contributions for a nondominated minimization front.

    Boundary points can be assigned infinity so the archive keeps both extremes,
    which is usually helpful for Pareto-front coverage and clean visualizations.
    """
    F = nondominated(F)
    n = len(F)
    if n == 0:
        return np.array([], dtype=float)
    if n <= 2:
        return np.full(n, np.inf, dtype=float)
    if ref is None:
        ref = _hypervolume_reference(F)
    ref = np.asarray(ref, dtype=float)
    F = F[np.argsort(F[:, 0], kind="mergesort")]
    x = F[:, 0]
    y = F[:, 1]
    contrib = np.empty(n, dtype=float)
    contrib[0] = max(0.0, x[1] - x[0]) * max(0.0, ref[1] - y[0])
    contrib[-1] = max(0.0, ref[0] - x[-1]) * max(0.0, y[-2] - y[-1])
    for i in range(1, n - 1):
        contrib[i] = max(0.0, x[i + 1] - x[i]) * max(0.0, y[i - 1] - y[i])
    if preserve_extremes:
        contrib[0] = np.inf
        contrib[-1] = np.inf
    return contrib


def prune_archive_hv(F: np.ndarray, max_size: int, preserve_extremes: bool = True) -> np.ndarray:
    """Hypervolume-contribution archive pruning for 2-objective minimization.

    It repeatedly removes the nondominated solution whose exclusive hypervolume
    contribution is smallest. This is more expensive than crowding distance but
    directly uses a quality indicator related to convergence + diversity.
    """
    F = nondominated(F)
    if len(F) <= max_size:
        return F
    while len(F) > max_size:
        ref = _hypervolume_reference(F)
        contrib = hypervolume_contributions_2d(F, ref=ref, preserve_extremes=preserve_extremes)
        remove_idx = int(np.argmin(contrib))
        F = np.delete(F, remove_idx, axis=0)
    return F[np.argsort(F[:, 0], kind="mergesort")]


def gd(F: np.ndarray, PF: np.ndarray) -> float:
    # Matches provided calculateGD.m: sqrt(sum(Dmin.^2)) / popSize
    if len(F) == 0:
        return float("inf")
    d = np.linalg.norm(F[:, None, :] - PF[None, :, :], axis=2)
    dmin = d.min(axis=1)
    return float(np.sqrt(np.sum(dmin ** 2)) / len(F))


def igd(F: np.ndarray, PF: np.ndarray) -> float:
    # Matches provided calculateIGD.m: sqrt(sum(Dmin.^2)) / pfSize
    if len(F) == 0:
        return float("inf")
    d = np.linalg.norm(PF[:, None, :] - F[None, :, :], axis=2)
    dmin = d.min(axis=1)
    return float(np.sqrt(np.sum(dmin ** 2)) / len(PF))


def roulette_select(ranks: np.ndarray, rng: np.random.Generator) -> int:
    weights = ranks.max() - ranks + 1e-12
    if weights.sum() <= 0 or not np.isfinite(weights).all():
        return int(rng.integers(0, len(ranks)))
    probs = weights / weights.sum()
    return int(rng.choice(len(ranks), p=probs))


def tournament_select(ranks: np.ndarray, rng: np.random.Generator, tournament_size: int) -> int:
    cands = rng.integers(0, len(ranks), size=max(2, tournament_size))
    best = cands[np.argmin(ranks[cands])]
    return int(best)


def tournament_select_crowding(ranks: np.ndarray, crowding: np.ndarray, rng: np.random.Generator, tournament_size: int) -> int:
    cands = rng.integers(0, len(ranks), size=max(2, tournament_size))
    best = cands[0]
    for c in cands[1:]:
        if ranks[c] < ranks[best]:
            best = c
        elif ranks[c] == ranks[best] and crowding[c] > crowding[best]:
            best = c
    return int(best)


def _downsample_front(F: np.ndarray, max_points: int) -> np.ndarray:
    F = nondominated(F)
    if len(F) <= max_points:
        return F
    order = np.argsort(F[:, 0], kind="mergesort")
    F = F[order]
    idx = np.unique(np.round(np.linspace(0, len(F) - 1, max_points)).astype(int))
    return F[idx]


def _record_history(cfg: GivenMOGAConfig, PF: np.ndarray, rows_metrics: list[dict[str, Any]], rows_fronts: list[dict[str, Any]], evals: int, generations: int, front: np.ndarray) -> None:
    if cfg.history_interval is None:
        return
    front = nondominated(front)
    rows_metrics.append({
        "evals": int(evals),
        "generations": int(generations),
        "front_size": int(len(front)),
        "gd": gd(front, PF),
        "igd": igd(front, PF),
        "spacing": spacing_metric(front),
        "hypervolume": hypervolume_2d(front),
    })
    sampled = _downsample_front(front, max(2, cfg.history_max_points))
    for i, (f1, f2) in enumerate(sampled):
        rows_fronts.append({
            "evals": int(evals),
            "generations": int(generations),
            "point_index": int(i),
            "f1": float(f1),
            "f2": float(f2),
        })


def run_given_moga(cfg: GivenMOGAConfig) -> GivenMOGAResult:
    if cfg.problem not in PROBLEMS:
        raise ValueError(f"Unsupported problem {cfg.problem}")
    if cfg.algorithm not in ALGORITHMS:
        raise ValueError(f"Unsupported algorithm {cfg.algorithm}. Choose from {ALGORITHMS}")
    rng = np.random.default_rng(cfg.seed)
    lower, upper = zdt_bounds(cfg.problem, cfg.dim)
    pop_size = cfg.pop_size if cfg.pop_size is not None else 10 * cfg.dim
    chrom_len = cfg.bits_per_var * cfg.dim
    PF = load_reference_front(cfg.problem)
    t0 = time.perf_counter()

    history_metrics: list[dict[str, Any]] = []
    history_fronts: list[dict[str, Any]] = []
    next_history_eval = 0 if cfg.history_interval is not None else None

    bits = rng.integers(0, 2, size=(pop_size, chrom_len), dtype=np.uint8)
    X = decode_bits(bits, cfg.bits_per_var, lower, upper)
    F = evaluate_zdt(X, cfg.problem)
    evals = pop_size
    generations = 0
    ranks = dominance_ranks(F)
    crowding = population_crowding(F, ranks)
    nds = nondominated(F[ranks == 1])
    archive = nds.copy()

    def active_front() -> np.ndarray:
        if cfg.algorithm == "moga_bonus":
            return archive
        if cfg.algorithm in {"moga_crowding", "moga_crowding_epsilon", "moga_crowding_hv"}:
            return archive
        return nds

    if cfg.history_interval is not None:
        _record_history(cfg, PF, history_metrics, history_fronts, evals, generations, active_front())
        next_history_eval = int(np.ceil(evals / cfg.history_interval) * cfg.history_interval)

    while evals + pop_size <= cfg.max_evals:
        generations += 1
        progress = evals / max(1, cfg.max_evals)
        new_bits = np.zeros_like(bits)
        start = 0

        improved = cfg.algorithm != "moga"
        crowding_based = cfg.algorithm in {"moga_crowding", "moga_crowding_epsilon", "moga_crowding_hv"}

        if improved:
            elite_count = max(1, int(round(cfg.elite_rate * pop_size)))
            if crowding_based:
                # Elitism uses the NSGA-II preference: lower dominance rank, then higher crowding distance.
                order = sorted(range(pop_size), key=lambda i: (ranks[i], -crowding[i]))
                order = np.asarray(order, dtype=int)
            else:
                order = np.argsort(ranks, kind="mergesort")
            new_bits[:elite_count] = bits[order[:elite_count]]
            start = elite_count
            # Adaptive mutation: high early exploration, low late disruption.
            p_mut = max(1.0 / chrom_len, cfg.p_mutation * (1.0 - progress))
        else:
            p_mut = cfg.p_mutation

        for i in range(start, pop_size):
            if crowding_based:
                p1 = tournament_select_crowding(ranks, crowding, rng, cfg.tournament_size)
                p2 = tournament_select_crowding(ranks, crowding, rng, cfg.tournament_size)
            elif cfg.algorithm == "moga_bonus":
                p1 = tournament_select(ranks, rng, cfg.tournament_size)
                p2 = tournament_select(ranks, rng, cfg.tournament_size)
            else:
                p1 = roulette_select(ranks, rng)
                p2 = roulette_select(ranks, rng)

            parent1 = bits[p1]
            parent2 = bits[p2]
            if rng.random() <= cfg.p_crossover:
                mask = rng.random(chrom_len) <= 0.5
                child = parent1.copy()
                child[~mask] = parent2[~mask]
            else:
                child = parent1.copy()

            mut_mask = rng.random(chrom_len) < p_mut
            child[mut_mask] = 1 - child[mut_mask]
            new_bits[i] = child

        bits = new_bits
        X = decode_bits(bits, cfg.bits_per_var, lower, upper)
        F = evaluate_zdt(X, cfg.problem)
        evals += pop_size
        ranks = dominance_ranks(F)
        crowding = population_crowding(F, ranks)
        current_nd = F[ranks == 1]
        nds = nondominated(np.vstack([nds, current_nd]))

        merged_archive = np.vstack([archive, current_nd])
        if cfg.algorithm == "moga_bonus":
            archive = prune_archive(merged_archive, cfg.archive_size)
        elif cfg.algorithm == "moga_crowding":
            archive = prune_archive_crowding(merged_archive, cfg.archive_size)
        elif cfg.algorithm == "moga_crowding_epsilon":
            archive = prune_archive_epsilon(merged_archive, cfg.archive_size, cfg.epsilon)
        elif cfg.algorithm == "moga_crowding_hv":
            archive = prune_archive_hv(merged_archive, cfg.archive_size, preserve_extremes=cfg.preserve_extremes)

        if cfg.history_interval is not None and next_history_eval is not None:
            if evals >= next_history_eval:
                _record_history(cfg, PF, history_metrics, history_fronts, evals, generations, active_front())
                next_history_eval += cfg.history_interval

        if cfg.verbose and generations % 50 == 0:
            print(f"  gen={generations} evals={evals} front={len(active_front())}")

    final_front = active_front()
    runtime = time.perf_counter() - t0
    return GivenMOGAResult(
        F=final_front,
        gd=gd(final_front, PF),
        igd=igd(final_front, PF),
        actual_evals=evals,
        generations=generations,
        front_size=len(final_front),
        runtime_sec=runtime,
        config=cfg,
        history_metrics=history_metrics,
        history_fronts=history_fronts,
    )
