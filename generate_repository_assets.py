from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
REPORT = ROOT / "report"
FIGURES.mkdir(exist_ok=True)
REPORT.mkdir(exist_ok=True)

DISPLAY_ORDER = ["MOGA", "MOGA Bonus", "Crowding", "Crowding + epsilon", "Crowding + HV"]
RANK_ORDER = ["MOGA Bonus", "Crowding + HV", "Crowding", "Crowding + epsilon", "MOGA"]
CASE_ORDER = [(p, d) for d in (10, 30) for p in ("ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6")]


def _save_bar(labels, values, title, ylabel, path, log=False):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(labels, values)
    if log:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_heatmap(summary, metric, title, path):
    rows = []
    for alg in DISPLAY_ORDER:
        vals = []
        for problem, dim in CASE_ORDER:
            hit = summary[(summary["algorithm"] == alg) & (summary["problem"] == problem) & (summary["dimension"] == dim)]
            vals.append(float(hit[metric].iloc[0]))
        rows.append(vals)
    data = np.asarray(rows, dtype=float)
    log_data = np.log10(data)
    fig, ax = plt.subplots(figsize=(13, 5.6))
    im = ax.imshow(log_data, aspect="auto")
    ax.set_title(title)
    ax.set_yticks(range(len(DISPLAY_ORDER)), DISPLAY_ORDER)
    labels = [f"{p} D={d}" for p, d in CASE_ORDER]
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.1e}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"log10({metric.upper()})")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generate_figures():
    ranking = pd.read_csv(RESULTS / "overall_ranking.csv")
    summary = pd.read_csv(RESULTS / "summary_full_comparison.csv")
    fronts = pd.read_csv(RESULTS / "front_size_summary.csv")
    sensitivity = pd.read_csv(RESULTS / "hyperparameter_sensitivity.csv")

    rank = ranking.set_index("Algorithm")
    _save_bar(
        RANK_ORDER,
        [rank.loc[a, "Mean IGD"] for a in RANK_ORDER],
        "Overall Mean IGD by Algorithm",
        "Mean IGD (log scale, lower is better)",
        FIGURES / "figure_1.png",
        log=True,
    )
    _save_bar(
        RANK_ORDER,
        [rank.loc[a, "Mean GD"] for a in RANK_ORDER],
        "Overall Mean GD by Algorithm",
        "Mean GD (log scale, lower is better)",
        FIGURES / "figure_2.png",
        log=True,
    )
    _save_heatmap(summary, "igd_mean", "IGD Heatmap Across Problems and Dimensions", FIGURES / "figure_3.png")
    _save_heatmap(summary, "gd_mean", "GD Heatmap Across Problems and Dimensions", FIGURES / "figure_4.png")

    front = fronts.set_index("Algorithm")
    _save_bar(
        DISPLAY_ORDER,
        [front.loc[a, "Mean final front size"] for a in DISPLAY_ORDER],
        "Mean Final Front Size by Algorithm",
        "Mean nondominated front size",
        FIGURES / "figure_5.png",
    )

    sens = sensitivity.set_index("Algorithm")
    _save_bar(
        DISPLAY_ORDER,
        [sens.loc[a, "Mean IGD"] for a in DISPLAY_ORDER],
        "Hyperparameter Sensitivity: Best Profile by Mean IGD",
        "Best profile mean IGD (log scale)",
        FIGURES / "figure_6.png",
        log=True,
    )


def _fmt(v):
    v = float(v)
    if v == 0:
        return "0"
    if abs(v) < 1e-3 or abs(v) >= 1e3:
        return f"{v:.3e}"
    return f"{v:.5f}".rstrip("0").rstrip(".")


def _table(data, widths=None, font_size=7.5):
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9ECEF")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B7B7B7")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _comparison_table(summary, metric, dim):
    header = ["Problem"] + DISPLAY_ORDER
    rows = [header]
    for problem in ("ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6"):
        row = [problem]
        for alg in DISPLAY_ORDER:
            hit = summary[(summary.algorithm == alg) & (summary.problem == problem) & (summary.dimension == dim)]
            row.append(_fmt(hit[metric].iloc[0]))
        rows.append(row)
    return rows


def generate_report():
    summary = pd.read_csv(RESULTS / "summary_full_comparison.csv")
    ranking = pd.read_csv(RESULTS / "overall_ranking.csv")
    sensitivity = pd.read_csv(RESULTS / "hyperparameter_sensitivity.csv")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, spaceAfter=8))
    styles.add(ParagraphStyle(name="SubCenter", parent=styles["Normal"], alignment=TA_CENTER, spaceAfter=5))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5, leading=11))

    doc = SimpleDocTemplate(
        str(REPORT / "MOGA_Study_Report.pdf"),
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
        title="MOGA Study Report",
        author="Ali Farrokhnejad",
    )
    story = []
    story += [
        Paragraph("MOGA Study Report", styles["TitleCenter"]),
        Paragraph("Evolutionary Multi-Objective Optimization", styles["Heading2"]),
        Paragraph("Prepared by: Ali Farrokhnejad", styles["SubCenter"]),
        Paragraph("Study date: 08 June 2026", styles["SubCenter"]),
        Spacer(1, 8),
        Paragraph("Abstract", styles["Heading1"]),
        Paragraph(
            "This report presents the implementation, evaluation, and extension of a Multi-Objective Genetic Algorithm (MOGA) on five standard ZDT benchmark problems: ZDT1, ZDT2, ZDT3, ZDT4, and ZDT6. Experiments cover D=10 and D=30 under a maximum budget of 200,000 function evaluations. The baseline MOGA is compared with tournament/elitist/archive MOGA, crowding-distance MOGA, crowding plus epsilon-grid archive control, and crowding plus hypervolume-contribution archive pruning. Archive-based and diversity-aware extensions substantially improve convergence and Pareto-front coverage over the baseline on most tested cases, while ZDT4 remains the most difficult benchmark because of its multimodal search landscape.",
            styles["BodyText"],
        ),
        Paragraph("1. Experimental Setup", styles["Heading1"]),
        _table([
            ["Item", "Setting"],
            ["Benchmark problems", "ZDT1, ZDT2, ZDT3, ZDT4, ZDT6"],
            ["Dimensions", "D=10 and D=30"],
            ["Evaluation budget", "Maximum 200,000 function evaluations"],
            ["Independent runs", "10 runs per problem, dimension, and algorithm setting"],
            ["Representation", "Binary-coded chromosome representation"],
            ["Primary metrics", "Generational Distance (GD) and Inverted Generational Distance (IGD)"],
            ["Supporting metrics", "Spacing, 2D hypervolume, front size, runtime, hyperparameter sensitivity"],
        ], [4.2 * cm, 12.2 * cm], 8),
        Spacer(1, 8),
        Paragraph("Lower GD and IGD are better. GD measures closeness to the reference Pareto front, while IGD emphasizes how completely the reference front is covered. The fixed evaluation budget prevents an improved method from quietly buying better numbers with additional search effort.", styles["BodyText"]),
        Paragraph("2. Algorithms Compared", styles["Heading1"]),
        _table([
            ["Algorithm", "Description"],
            ["MOGA", "Baseline binary-coded MOGA with roulette-style selection, uniform crossover, bit mutation, and dominance-based ranking."],
            ["MOGA Bonus", "Tournament selection, elitism, adaptive mutation, and an external nondominated archive."],
            ["Crowding", "NSGA-II-style crowding distance in selection and archive pruning."],
            ["Crowding + epsilon", "Crowding with epsilon-grid filtering to control near-duplicate archive points."],
            ["Crowding + HV", "Crowding with hypervolume-contribution archive pruning."],
        ], [4.0 * cm, 12.4 * cm], 7.5),
        PageBreak(),
        Paragraph("3. Baseline MOGA Results", styles["Heading1"]),
    ]

    for dim in (10, 30):
        for metric, label in (("gd_mean", "GD"), ("igd_mean", "IGD")):
            rows = [["Problem", "MOGA"]]
            for p in ("ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6"):
                v = summary[(summary.algorithm == "MOGA") & (summary.problem == p) & (summary.dimension == dim)][metric].iloc[0]
                rows.append([p, _fmt(v)])
            story += [Paragraph(f"{label} at D={dim}", styles["Heading2"]), _table(rows, [6 * cm, 5 * cm], 8), Spacer(1, 7)]

    story += [PageBreak(), Paragraph("4. Extended Comparison", styles["Heading1"])]
    for dim in (10, 30):
        for metric, label in (("gd_mean", "GD"), ("igd_mean", "IGD")):
            story += [Paragraph(f"{label} at D={dim}", styles["Heading2"]), _table(_comparison_table(summary, metric, dim), [2.1 * cm] + [2.85 * cm] * 5, 6.6), Spacer(1, 8)]

    story += [PageBreak(), Paragraph("5. Visual Analysis", styles["Heading1"])]
    captions = [
        "Overall mean IGD by algorithm. Lower values indicate better average Pareto-front coverage.",
        "Overall mean GD by algorithm. Lower values indicate closer convergence to the reference front.",
        "IGD heatmap across all problems and dimensions.",
        "GD heatmap across all problems and dimensions. ZDT4 dominates the difficulty pattern.",
        "Mean final nondominated front size. Archive-based methods retain substantially more solutions than the baseline.",
        "Hyperparameter sensitivity summary using the best tested profile by mean IGD for each algorithm.",
    ]
    for idx, cap in enumerate(captions, 1):
        img = FIGURES / f"figure_{idx}.png"
        story += [Image(str(img), width=17 * cm, height=9.4 * cm), Paragraph(f"Figure {idx}. {cap}", styles["Small"]), Spacer(1, 8)]
        if idx in {2, 4}:
            story.append(PageBreak())

    story += [PageBreak(), Paragraph("6. Overall Ranking and Sensitivity", styles["Heading1"])]
    rank_rows = [["Rank", "Algorithm", "Mean IGD", "Median IGD", "Mean GD", "Mean front size"]]
    for _, r in ranking.iterrows():
        rank_rows.append([str(int(r["Rank"])), r["Algorithm"], _fmt(r["Mean IGD"]), _fmt(r["Median IGD"]), _fmt(r["Mean GD"]), _fmt(r["Mean front size"])])
    story += [_table(rank_rows, [1.1 * cm, 3.5 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 3 * cm], 7.2), Spacer(1, 10)]

    sens_rows = [["Algorithm", "Best profile", "Mean IGD", "Mean GD"]]
    for _, r in sensitivity.iterrows():
        sens_rows.append([r["Algorithm"], r["Best profile by mean IGD"], _fmt(r["Mean IGD"]), _fmt(r["Mean GD"])])
    story += [_table(sens_rows, [4 * cm, 4.5 * cm, 3.2 * cm, 3.2 * cm], 7.5), Spacer(1, 10)]

    story += [
        Paragraph("7. Discussion", styles["Heading1"]),
        Paragraph("The baseline MOGA completes the benchmark protocol but produces much weaker GD and IGD than the archive-based variants. Its mean final front contains only about 20.85 nondominated solutions, compared with hundreds for the archive-based methods, which helps explain the poorer IGD and incomplete front coverage.", styles["BodyText"]),
        Paragraph("MOGA Bonus has the best overall mean IGD. Tournament selection provides more consistent selection pressure than roulette-style selection; elitism protects strong solutions; adaptive mutation balances exploration with refinement; and the external archive prevents nondominated solutions from disappearing between generations.", styles["BodyText"]),
        Paragraph("Crowding and Crowding + HV have especially strong typical performance. Crowding distance is a defensible diversity mechanism because it improves distribution without changing the optimization objective itself. Epsilon filtering reduces near-duplicates but can remove nearby points that still benefit IGD. Hypervolume-contribution pruning is more indicator-driven and can preserve high-value points, although its behavior depends on the reference point.", styles["BodyText"]),
        Paragraph("ZDT4 remains the dominant failure case. Its multimodal structure creates local optima away from the true Pareto front, so it exposes the limits of all tested variants more clearly than the easier ZDT cases.", styles["BodyText"]),
        Paragraph("8. Conclusion", styles["Heading1"]),
        Paragraph("Across the tested ZDT suite, the strongest overall mean IGD is achieved by MOGA Bonus, while crowding-based variants achieve very strong median IGD and diversity. Relative to the baseline aggregate mean IGD of 1.512730, MOGA Bonus reaches 0.108612, an improvement of approximately 92.8%. The results support archive and diversity mechanisms as substantial improvements, but they do not justify a universal claim that one variant dominates every benchmark.", styles["BodyText"]),
        Paragraph("Reproducibility", styles["Heading1"]),
        Paragraph("The repository contains the implementation, benchmark runners, tuning utilities, archived result tables, visualization generator, and this report. The figures and PDF are regenerated from the committed numeric results by GitHub Actions so the public artifact remains internally reproducible.", styles["BodyText"]),
    ]

    doc.build(story)


def main():
    generate_figures()
    generate_report()
    print("Generated six figures and report/MOGA_Study_Report.pdf")


if __name__ == "__main__":
    main()
