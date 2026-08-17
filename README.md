# Evolutionary Multi-Objective Optimization with MOGA

A reproducible Python study of **Multi-Objective Genetic Algorithm (MOGA)** variants on the ZDT benchmark suite, focused on convergence, Pareto-front coverage, diversity preservation, and archive design.

The study evaluates **ZDT1, ZDT2, ZDT3, ZDT4, and ZDT6** at **10 and 30 dimensions** under a fixed maximum budget of **200,000 function evaluations**, with **10 independent runs** per problem/configuration in the full experiment.

> **Provenance:** the baseline MOGA structure was supplied as the starting algorithm. The Python reimplementation, experiment runner, benchmarking/evaluation pipeline, improvement modes, tuning, analysis, visualizations, and reproducibility tooling in this repository were developed for the study.

## What was explored

Five algorithm modes are compared:

| Mode | Main idea |
|---|---|
| `moga` | Baseline binary-coded MOGA with roulette-style selection, uniform crossover, bit mutation, and dominance-based ranking |
| `moga_bonus` | Tournament selection, elitism, adaptive mutation, and an external nondominated archive |
| `moga_crowding` | NSGA-II-style crowding distance for selection and archive pruning |
| `moga_crowding_epsilon` | Crowding plus epsilon-grid archive filtering |
| `moga_crowding_hv` | Crowding plus hypervolume-contribution archive pruning |

The primary metrics are **Generational Distance (GD)** and **Inverted Generational Distance (IGD)**. Supporting analysis includes spacing, 2D hypervolume, front size, runtime, and hyperparameter sensitivity.

## Key result

Across the complete problem/dimension comparison, the strongest overall mean IGD was obtained by the tournament/elitist/archive variant:

| Algorithm | Mean IGD | Mean GD | Mean front size |
|---|---:|---:|---:|
| MOGA Bonus | **0.108612** | **0.143393** | 699.84 |
| Crowding + HV | 0.125987 | 0.201875 | 645.05 |
| Crowding | 0.125987 | 0.201978 | 645.06 |
| Crowding + epsilon | 0.126006 | 0.209144 | 274.07 |
| Baseline MOGA | 1.512730 | 13.735000 | 20.85 |

This corresponds to an approximately **92.8% reduction in aggregate mean IGD** for MOGA Bonus relative to the baseline. The largest remaining difficulty is ZDT4, whose multimodal landscape remains substantially harder than the other tested problems.

## Repository structure

```text
.
├── src/
│   └── moga.py                 # ZDT problems, MOGA variants, metrics, archives
├── run_experiments.py          # full/reduced benchmark runner
├── tune_hyperparams.py         # parameter sensitivity/tuning
├── make_visualizations.py      # result plots
├── generate_report.py          # report-generation helper
├── results/                    # summary and comparison CSVs
├── figures/                    # generated analysis figures
├── report/
│   └── MOGA_Study_Report.pdf   # complete study report
└── docs/
    └── algorithm_modes.md
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the full comparison

```bash
python run_experiments.py \
  --max-evals 200000 \
  --n-runs 10 \
  --algorithms moga moga_bonus moga_crowding moga_crowding_epsilon moga_crowding_hv \
  --workers 30 \
  --save-fronts \
  --out-dir results/reproduced_runs
```

Use a smaller `--workers` value on machines with fewer CPU cores.

A lightweight check can be run with a lower evaluation budget:

```bash
python run_experiments.py \
  --problems ZDT1 \
  --dims 10 \
  --algorithms moga moga_bonus \
  --max-evals 2000 \
  --n-runs 1 \
  --workers 1 \
  --out-dir results/smoke_test
```

## Experimental design

- Benchmark problems: ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
- Dimensions: D=10 and D=30
- Maximum function evaluations: 200,000
- Full-study repetitions: 10 independent runs per configuration
- Primary metrics: GD and IGD
- Additional metrics: spacing, 2D hypervolume, front size, runtime
- Parallel execution support for independent runs

The fixed evaluation budget is enforced so algorithm comparisons are made under the same computational search allowance.

When the original supplied Pareto-front text files are unavailable, the implementation generates analytical ZDT reference fronts so the repository remains standalone. The archived CSVs and report are the authoritative record of the original full experiment.

## Interpretation

The improvements are not uniformly interchangeable. Tournament selection, elitism, adaptive mutation, and external archiving give the strongest overall mean IGD in these experiments. Crowding-distance variants provide particularly strong typical performance and diversity preservation, while epsilon-grid and hypervolume pruning expose trade-offs in archive size, front density, and indicator-driven selection. ZDT4 remains the main failure case and prevents a simplistic claim that one extension solves every benchmark equally well.

## Data

The repository contains implementation code and generated benchmark outputs only. No external datasets are required because the ZDT objective functions are generated analytically.
