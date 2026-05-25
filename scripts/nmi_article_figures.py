"""Generate Nature-style consolidated figures for the NMI manuscript.

The figures are built from cached CSV outputs in ``outputs/``.  Text remains
editable in SVG/PDF outputs; no generated raster text is used.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
MM = 1 / 25.4

PALETTE = {
    "benchmark": "#0072B2",  # blue
    "certified": "#009E73",  # bluish green
    "ambiguous": "#E69F00",  # orange
    "false": "#D55E00",  # vermillion
    "oracle": "#000000",
    "gray": "#8C8C8C",
    "light_gray": "#E6E8EB",
    "pale_blue": "#D9EAF7",
    "pale_green": "#DDF1EA",
    "pale_orange": "#F8E7BF",
    "dark": "#222222",
}

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 6.5,
        "axes.titlesize": 7.0,
        "axes.labelsize": 6.7,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6.0,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "lines.linewidth": 1.15,
        "lines.markersize": 3.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def pct(x):
    return 100.0 * np.asarray(x, dtype=float)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
    )


def clean(ax: plt.Axes, *, xgrid: bool = False, ygrid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ygrid:
        ax.grid(axis="y", color="#D9DDE2", lw=0.5, alpha=0.75)
    if xgrid:
        ax.grid(axis="x", color="#D9DDE2", lw=0.5, alpha=0.75)
    ax.set_axisbelow(True)


def add_box(ax, xy, w, h, text, fc, ec=None, fs=6.2, weight="normal"):
    if ec is None:
        ec = PALETTE["gray"]
    box = patches.FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=0.7,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight=weight,
    )
    return box


def arrow(ax, start, end, color="#555555", lw=0.8, style="-|>"):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle=style, lw=lw, color=color, shrinkA=2, shrinkB=2),
    )


def certification_standard() -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(183 * MM, 104 * MM),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "width_ratios": [1.15, 1.0]},
    )
    axes = axes.ravel()

    ax = axes[0]
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Current benchmark logic", loc="left", pad=2)
    add_box(ax, (0.04, 0.49), 0.24, 0.18, "benchmark\nscore", PALETTE["pale_blue"], PALETTE["benchmark"])
    add_box(ax, (0.38, 0.49), 0.24, 0.18, "model/candidate\nchoice", "white")
    add_box(ax, (0.72, 0.49), 0.24, 0.18, "deployment\naction", "white")
    arrow(ax, (0.28, 0.58), (0.38, 0.58))
    arrow(ax, (0.62, 0.58), (0.72, 0.58))
    add_box(
        ax,
        (0.36, 0.20),
        0.28,
        0.13,
        "deployment response\nunobserved",
        "#FFF2E1",
        PALETTE["false"],
        fs=5.8,
        weight="bold",
    )
    arrow(ax, (0.50, 0.49), (0.50, 0.33), color=PALETTE["false"], lw=0.8)
    panel_label(ax, "a")

    ax = axes[1]
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Certification logic", loc="left", pad=2)
    add_box(ax, (0.03, 0.62), 0.24, 0.15, "benchmark\nevidence", PALETTE["pale_blue"], PALETTE["benchmark"])
    add_box(ax, (0.03, 0.34), 0.24, 0.15, "declared\ndeployment claim", "white")
    add_box(ax, (0.38, 0.48), 0.25, 0.18, "certification\ndiagnostic", PALETTE["pale_green"], PALETTE["certified"])
    add_box(ax, (0.73, 0.68), 0.23, 0.13, "certified\nviable", PALETTE["pale_green"], PALETTE["certified"])
    add_box(ax, (0.73, 0.48), 0.23, 0.13, "certified\nnonviable", "white", PALETTE["gray"])
    add_box(ax, (0.73, 0.26), 0.23, 0.13, "ambiguous:\nacquire response", PALETTE["pale_orange"], PALETTE["ambiguous"])
    arrow(ax, (0.27, 0.69), (0.38, 0.58))
    arrow(ax, (0.27, 0.41), (0.38, 0.55))
    arrow(ax, (0.63, 0.57), (0.73, 0.74), PALETTE["certified"])
    arrow(ax, (0.63, 0.57), (0.73, 0.55))
    arrow(ax, (0.63, 0.55), (0.73, 0.33), PALETTE["ambiguous"])
    panel_label(ax, "b")

    ax = axes[2]
    ax.set_xlim(-0.2, 3.2)
    ax.set_ylim(-0.3, 2.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Response-rank geometry", loc="left", pad=2)
    plane = patches.Polygon(
        [[0.0, 0.15], [2.55, 0.45], [3.0, 1.35], [0.45, 1.05]],
        closed=True,
        facecolor=PALETTE["pale_blue"],
        edgecolor=PALETTE["benchmark"],
        lw=0.8,
        alpha=0.85,
    )
    ax.add_patch(plane)
    ax.text(2.5, 0.28, "benchmark span", fontsize=6.0, color=PALETTE["benchmark"])
    origin = np.array([0.35, 0.35])
    probe1 = np.array([1.55, 0.55])
    probe2 = np.array([0.88, 1.05])
    proj = np.array([1.85, 0.80])
    kstar = np.array([2.35, 2.20])
    ax.annotate("", xy=probe1, xytext=origin, arrowprops=dict(arrowstyle="-|>", lw=0.85, color=PALETTE["benchmark"], alpha=0.9))
    ax.annotate("", xy=probe2, xytext=origin, arrowprops=dict(arrowstyle="-|>", lw=0.85, color=PALETTE["benchmark"], alpha=0.9))
    ax.text(probe1[0] + 0.02, probe1[1] - 0.15, r"$k_1$", fontsize=5.7, color=PALETTE["benchmark"])
    ax.text(probe2[0] - 0.16, probe2[1] + 0.02, r"$k_2$", fontsize=5.7, color=PALETTE["benchmark"])
    ax.annotate("", xy=proj, xytext=origin, arrowprops=dict(arrowstyle="-|>", lw=1.1, color=PALETTE["benchmark"]))
    ax.annotate("", xy=kstar, xytext=origin, arrowprops=dict(arrowstyle="-|>", lw=1.1, color=PALETTE["ambiguous"]))
    ax.plot(
        [proj[0], kstar[0]],
        [proj[1], kstar[1]],
        ls=(0, (2, 2)),
        color=PALETTE["ambiguous"],
        lw=1.15,
        dash_capstyle="butt",
        solid_capstyle="butt",
    )
    span_dir = probe1 - origin
    span_dir = span_dir / np.linalg.norm(span_dir) * 0.12
    resid_dir = kstar - proj
    resid_dir = resid_dir / np.linalg.norm(resid_dir) * 0.12
    corner = np.vstack([proj + span_dir, proj + span_dir + resid_dir, proj + resid_dir])
    ax.plot(corner[:, 0], corner[:, 1], color=PALETTE["gray"], lw=0.65)
    ax.text(kstar[0] + 0.05, kstar[1], r"$k_\star$", fontsize=7.0, color=PALETTE["ambiguous"])
    ax.text(proj[0] + 0.07, proj[1] - 0.30, r"$P_B k_\star$", fontsize=6.0, color=PALETTE["benchmark"])
    ax.text(2.05, 1.45, r"$r_\star=(I-P_B)k_\star$", fontsize=6.0, color=PALETTE["ambiguous"])
    ax.text(0.10, 2.45, r"$g=\|r_\star\|$", fontsize=7.0, color=PALETTE["dark"])
    panel_label(ax, "c")

    ax = axes[3]
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.05, 3.15)
    ax.set_yticks([2.45, 1.55, 0.65], ["certified\nviable", "ambiguous", "certified\nnonviable"])
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xlabel("deployment response")
    ax.axvline(0.50, color=PALETTE["dark"], lw=0.8)
    ax.text(0.51, 2.95, "threshold", ha="left", va="top", fontsize=5.8)
    intervals = [
        (0.66, 0.08, 2.45, PALETTE["certified"], "check"),
        (0.40, 0.20, 1.55, PALETTE["ambiguous"], "open"),
        (0.22, 0.10, 0.65, PALETTE["gray"], "x"),
    ]
    for centre, half, y, color, marker in intervals:
        ax.hlines(y, centre - half, centre + half, color=color, lw=3.0)
        ax.plot(centre, y, marker="o", ms=4.2, color=color, mfc="white" if marker == "open" else color)
    clean(ax, ygrid=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Candidate-level certificate", loc="left", pad=2)
    panel_label(ax, "d")

    save(fig, "nmi_certification_standard")


def main_results() -> None:
    coverage_summary = pd.read_csv(OUT / "revision_conformal_repeated_summary.csv")
    coverage_runs = pd.read_csv(OUT / "revision_conformal_repeated_runs.csv")
    coverage = dict(zip(coverage_summary["metric"], coverage_summary["mean"]))
    leaderboard = pd.read_csv(OUT / "revision_leaderboard_repeated_summary.csv")
    leaderboard = leaderboard[(leaderboard["setting"] == "structured") & (leaderboard["model"].str.len() == 1)].copy()
    vision_by_model = pd.read_csv(OUT / "vision_robustness_response_summary_by_model.csv")
    vision = pd.read_csv(OUT / "vision_robustness_response_summary.csv").iloc[0]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(183 * MM, 110 * MM),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "width_ratios": [1.0, 1.0]},
    )
    axes = axes.ravel()

    ax = axes[0]
    metrics = [
        ("benchmark_conformal_coverage", "benchmark\nchannel", PALETTE["benchmark"]),
        ("benchmark_transfer_coverage", "transfer to\ndeployment", PALETTE["false"]),
        ("oracle_deployment_coverage", "oracle\ndeployment", "white"),
        ("response_rank_coverage", "response-rank\ninterval", PALETTE["certified"]),
    ]
    rng = np.random.default_rng(7)
    for i, (metric, label, color) in enumerate(metrics):
        vals = pct(coverage_runs[metric])
        jitter = rng.uniform(-0.11, 0.11, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=7, color=color if color != "white" else "none",
                   edgecolor=PALETTE["oracle"] if color == "white" else color, alpha=0.35, lw=0.45)
        mean = pct(coverage[metric])
        lo, hi = np.percentile(vals, [2.5, 97.5])
        ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], fmt="o", ms=4.8,
                    color=PALETTE["oracle"] if color == "white" else color,
                    mfc="white" if color == "white" else color, capsize=2.0, lw=0.8)
        ax.text(i, min(mean + 9.0, 103), f"{mean:.1f}", ha="center", fontsize=5.8)
    ax.axhline(95, color=PALETTE["dark"], lw=0.75, ls=(0, (3, 2)))
    ax.text(3.05, 96.3, "95%", ha="left", va="bottom", fontsize=5.6, color=PALETTE["dark"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("coverage (%)")
    ax.set_xticks(range(4), [m[1] for m in metrics])
    ax.set_title("Coverage belongs to a response channel", loc="left", pad=2)
    clean(ax)
    panel_label(ax, "a")

    ax = axes[1]
    ax.scatter(coverage_runs["deployment_residual"], pct(coverage_runs["benchmark_transfer_coverage"]),
               s=12, color=PALETTE["false"], alpha=0.65, label="benchmark-calibrated transfer")
    ax.scatter(coverage_runs["deployment_residual"], pct(coverage_runs["response_rank_coverage"]),
               s=12, color=PALETTE["certified"], alpha=0.65, label="response-rank interval")
    for col, color in [("benchmark_transfer_coverage", PALETTE["false"]), ("response_rank_coverage", PALETTE["certified"])]:
        x = coverage_runs["deployment_residual"].to_numpy()
        y = pct(coverage_runs[col].to_numpy())
        order = np.argsort(x)
        bins = np.array_split(order, 6)
        ax.plot([x[b].mean() for b in bins], [y[b].mean() for b in bins], color=color, lw=1.5)
    ax.axhline(95, color=PALETTE["dark"], lw=0.75, ls=(0, (3, 2)))
    ax.text(coverage_runs["deployment_residual"].min(), 96.3, "95%", ha="left", va="bottom", fontsize=5.6, color=PALETTE["dark"])
    ax.set_xlabel(r"residual size $g$")
    ax.set_ylabel("deployment coverage (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Residual directions collapse transfer", loc="left", pad=2)
    ax.legend(frameon=False, loc="lower left", handlelength=1.0)
    clean(ax)
    panel_label(ax, "b")

    ax = axes[2]
    models = leaderboard["model"].to_list()
    mae = leaderboard["mean_benchmark_mae"].to_numpy()
    cert = pct(leaderboard["mean_certified_fraction"].to_numpy())
    left_rank = pd.Series(mae).rank(method="first", ascending=True).to_numpy()
    right_rank = pd.Series(cert).rank(method="first", ascending=False).to_numpy()
    for i, model in enumerate(models):
        color = PALETTE["false"] if model == "A" else (PALETTE["certified"] if model == "E" else PALETTE["gray"])
        ax.plot([0, 1], [left_rank[i], right_rank[i]], color=color, lw=1.3, alpha=0.95)
        ax.scatter([0, 1], [left_rank[i], right_rank[i]], color=color, s=18, zorder=3)
        ax.text(-0.04, left_rank[i], f"{model}", ha="right", va="center", fontsize=6.2, color=color)
        ax.text(1.04, right_rank[i], f"{cert[i]:.0f}%", ha="left", va="center", fontsize=6.2, color=color)
    ax.set_xlim(-0.24, 1.27)
    ax.set_ylim(5.45, 0.55)
    ax.set_xticks([0, 1], ["benchmark\nMAE rank", "certified\ntop-100 yield"])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylabel("rank (1 = best)")
    ax.set_title("Best score can be the worst certifier", loc="left", pad=2)
    clean(ax, ygrid=False)
    panel_label(ax, "c")

    ax = axes[3]
    y = np.arange(len(vision_by_model))[::-1]
    clean_vals = pct(vision_by_model["mean_clean_accuracy"].to_numpy())
    robust_vals = pct(vision_by_model["mean_robust_suite_accuracy"].to_numpy())
    for yi, c, r in zip(y, clean_vals, robust_vals):
        ax.plot([c, r], [yi, yi], color=PALETTE["light_gray"], lw=1.8, zorder=1)
    ax.scatter(clean_vals, y, color=PALETTE["benchmark"], s=18, label="clean accuracy", zorder=3)
    ax.scatter(robust_vals, y, color=PALETTE["ambiguous"], s=18, label="corruption robustness", zorder=3)
    ax.set_yticks(y, vision_by_model["model"].str.replace("_", " "))
    ax.set_xlabel("accuracy / robustness (%)")
    ax.set_xlim(0, 103)
    ax.text(
        0.02,
        0.06,
        f"winner changes in {pct(vision['clean_choice_changed_for_robustness_fraction']):.0f}% of splits",
        transform=ax.transAxes,
        fontsize=6.0,
        color=PALETTE["dark"],
    )
    ax.set_title("Clean winners need not be robust winners", loc="left", pad=2)
    ax.legend(frameon=False, loc="lower right", handlelength=1.0)
    clean(ax, xgrid=True, ygrid=False)
    panel_label(ax, "d")

    save(fig, "nmi_main_results")


def public_audits() -> None:
    vision = pd.read_csv(OUT / "vision_robustness_response_summary.csv").iloc[0]
    tox = pd.read_csv(OUT / "tox21_response_certification_summary.csv").iloc[0]
    spin = pd.read_csv(OUT / "ncs_public_spin_defect_summary.csv").iloc[0]
    mat = pd.read_csv(OUT / "matbench_discovery_certification_summary.csv").iloc[0]
    jar = pd.read_csv(OUT / "jarvis_public_model_certification_summary.csv").iloc[0]
    mat_sens = pd.read_csv(OUT / "matbench_discovery_fiber_sensitivity_summary.csv")
    jar_sens = pd.read_csv(OUT / "jarvis_fiber_sensitivity_summary.csv")
    tox_audit = pd.read_csv(OUT / "tox21_response_certification_audit.csv")

    fig = plt.figure(figsize=(183 * MM, 126 * MM), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.22, 1.0], width_ratios=[1.0, 1.0, 1.15])
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    rows = ["Vision", "Tox21", "Spin defects", "Matbench", "JARVIS"]
    cols = ["Benchmark observable", "Deployment claim", "Fiber rule", "Ambiguous", "Next response"]
    cells = [
        ["clean prediction", "corruption robust", "finite prediction", "22.7%", "corruptions"],
        ["7 assays", "SR-p53", "exact assay pattern", "97.9%", "SR-p53"],
        ["bare host", "substrate viability", "reported substrates", "23.0%", "substrate"],
        ["formation energy", "stability", "quantile / kNN / window", "100%", "stability"],
        ["formation energy", "band gap", "quantile / kNN / window", "100%", "band gap"],
    ]
    y0 = 0.76
    row_h = 0.145
    xpos = [0.03, 0.24, 0.42, 0.59, 0.76, 0.91]
    headers = ["domain", *cols]
    for xh, head in zip(xpos, headers):
        ax.text(xh, 0.91, head, weight="bold", fontsize=6.35, ha="left" if head == "domain" else "center", va="center")
    for r, row in enumerate(rows):
        ybase = y0 - r * row_h
        fc = "#F7F8FA" if r % 2 == 0 else "white"
        ax.add_patch(patches.Rectangle((0.0, ybase - 0.060), 0.99, 0.115, facecolor=fc, edgecolor="#E2E4E7", lw=0.4))
        ax.text(xpos[0], ybase, row, fontsize=6.4, ha="left", va="center")
        for j, txt in enumerate(cells[r]):
            color = PALETTE["ambiguous"] if j == 3 else PALETTE["dark"]
            weight = "bold" if j == 3 else "normal"
            ax.text(xpos[j + 1], ybase, txt, fontsize=5.95, ha="center", va="center", color=color, fontweight=weight)
    ax.set_title("Cross-domain evidence map", loc="left", pad=2)
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[1, 0])
    names = ["Vision", "Tox21", "Spin defects", "Matbench", "JARVIS"]
    ambiguous = np.array([
        pct(vision["median_ambiguous_fraction"]),
        pct(tox["ambiguous_candidate_fraction"]),
        pct(spin["response_ambiguous"] / spin["hosts_with_valid_heterostructure_t2"]),
        pct(mat["median_ambiguous_fraction"]),
        pct(jar["median_ambiguous_fraction"]),
    ], dtype=float)
    order = np.argsort(ambiguous)
    y = np.arange(len(names))
    ax.hlines(y, 0, ambiguous[order], color=PALETTE["ambiguous"], lw=2.4)
    ax.scatter(ambiguous[order], y, color=PALETTE["ambiguous"], s=28, zorder=3)
    ax.set_yticks(y, np.array(names)[order])
    ax.set_xlim(0, 105)
    ax.set_xlabel("ambiguous candidates (%)")
    for yi, val in zip(y, ambiguous[order]):
        ax.text(val + 2, yi, f"{val:.1f}%", va="center", fontsize=5.8)
    ax.set_title("Released responses leave ambiguity", loc="left", pad=2)
    clean(ax, xgrid=True, ygrid=False)
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 1])
    methods = ["quantile_20", "knn_25", "interval_mae", "interval_q80_abs_error"]
    labels = ["quantile", "kNN", "MAE\nwindow", "80% error\nwindow"]
    mat_lookup = dict(zip(mat_sens["method"], mat_sens["median_certifiable_fraction"]))
    jar_lookup = dict(zip(jar_sens["method"], jar_sens["median_certifiable_fraction"]))
    mat_raw = pct([mat_lookup[m] for m in methods])
    jar_raw = pct([jar_lookup[m] for m in methods])
    floor = 0.004
    mat_vals = np.maximum(mat_raw, floor)
    jar_vals = np.maximum(jar_raw, floor)
    x = np.arange(len(methods))
    ax.scatter(x - 0.08, mat_vals, color=PALETTE["benchmark"], s=26, label="Matbench")
    ax.scatter(x + 0.08, jar_vals, color=PALETTE["ambiguous"], s=26, label="JARVIS")
    for vals, raw, offset, color in [(mat_vals, mat_raw, -0.08, PALETTE["benchmark"]), (jar_vals, jar_raw, 0.08, PALETTE["ambiguous"])]:
        zero = np.asarray(raw) == 0
        if zero.any():
            ax.scatter(
                x[zero] + offset,
                vals[zero],
                marker="v",
                s=30,
                facecolors="white",
                edgecolors=color,
                lw=0.8,
                zorder=3,
            )
    for xi, m, j in zip(x, mat_vals, jar_vals):
        ax.vlines(xi, min(m, j), max(m, j), color=PALETTE["light_gray"], lw=1.6, zorder=0)
    ax.set_yscale("log")
    ax.set_ylim(0.002, 20)
    ax.set_xticks(x, labels)
    ax.set_ylabel("certifiable candidates (%)")
    ax.set_title("Continuous-fiber sensitivity", loc="left", pad=2)
    ax.legend(frameon=False, loc="upper left", handlelength=1.0)
    ax.text(
        0.04,
        0.14,
        "open triangles: 0% median\nplotted at floor",
        transform=ax.transAxes,
        fontsize=5.2,
        ha="left",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=1.0),
    )
    clean(ax)
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 2])
    tox_audit = tox_audit.copy()
    tox_audit["total"] = tox_audit["deployment_inactive"] + tox_audit["deployment_active"]
    tox_audit["active_frac"] = tox_audit["deployment_active"] / tox_audit["total"]
    tox_audit["balance"] = (tox_audit["active_frac"] - 0.5).abs()
    example = (
        tox_audit[(tox_audit["certification_class"] == "response_ambiguous") & (tox_audit["total"] >= 2)]
        .sort_values(["balance", "total"])
        .head(5)
        .copy()
    )
    y = np.arange(len(example))[::-1]
    inactive = example["deployment_inactive"].to_numpy()
    active = example["deployment_active"].to_numpy()
    total = inactive + active
    ax.barh(y, inactive / total * 100, color=PALETTE["light_gray"], label="inactive")
    ax.barh(y, active / total * 100, left=inactive / total * 100, color=PALETTE["false"], label="active")
    ax.set_yticks(y, example["benchmark_pattern"].str.replace("|", "", regex=False))
    ax.set_xlim(0, 100)
    ax.set_xlabel("SR-p53 labels within same benchmark fiber (%)")
    ax.set_title("Same benchmark evidence, mixed deployment labels", loc="left", pad=2)
    for yi, n, a in zip(y, total, active):
        ax.text(101, yi, f"{int(a)} active / {int(n-a)} inactive", va="center", fontsize=5.5)
    ax.text(
        0.02,
        0.05,
        "both outcomes in each fiber",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.7,
        color=PALETTE["dark"],
    )
    ax.legend(frameon=False, loc="upper left", handlelength=1.0)
    clean(ax, xgrid=True, ygrid=False)
    panel_label(ax, "d")

    save(fig, "nmi_public_audits")


def operational_replay() -> None:
    tox = pd.read_csv(OUT / "tox21_operational_replay_summary.csv").set_index("policy")
    jar = pd.read_csv(OUT / "jarvis_operational_replay_summary.csv").set_index("policy")
    cost_curve = pd.read_csv(OUT / "operational_cost_curve.csv")
    break_even = pd.read_csv(OUT / "operational_cost_break_even_summary.csv").set_index("domain")
    choices = pd.read_csv(OUT / "jarvis_operational_model_choice.csv")
    case = pd.read_csv(OUT / "jarvis_operational_case_study.csv")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(183 * MM, 116 * MM),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "width_ratios": [1.0, 1.0]},
    )
    axes = axes.ravel()

    def flow_panel(ax, domain, df, baseline, cert, acquire, false_text):
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{domain} decision flow", loc="left", pad=2)
        add_box(ax, (0.03, 0.42), 0.20, 0.17, "held-out\ncandidates", "white")
        add_box(ax, (0.35, 0.67), 0.23, 0.16, "benchmark\naction", PALETTE["pale_blue"], PALETTE["benchmark"])
        add_box(ax, (0.72, 0.67), 0.23, 0.16, false_text, "#FFF2E1", PALETTE["false"])
        add_box(ax, (0.35, 0.36), 0.23, 0.16, f"certify now\n{pct(df.loc[cert, 'mean_certified_fraction']):.2f}%", PALETTE["pale_green"], PALETTE["certified"])
        add_box(ax, (0.35, 0.10), 0.23, 0.16, "ambiguous:\nmeasure", PALETTE["pale_orange"], PALETTE["ambiguous"])
        add_box(ax, (0.72, 0.28), 0.23, 0.16, f"after acquisition\n{pct(df.loc[acquire, 'mean_error_rate_all_candidates']):.3f}% false", PALETTE["pale_green"], PALETTE["certified"])
        arrow(ax, (0.23, 0.51), (0.35, 0.75), PALETTE["benchmark"])
        arrow(ax, (0.58, 0.75), (0.72, 0.75), PALETTE["false"])
        arrow(ax, (0.23, 0.50), (0.35, 0.44), PALETTE["certified"])
        arrow(ax, (0.23, 0.46), (0.35, 0.18), PALETTE["ambiguous"])
        arrow(ax, (0.58, 0.44), (0.72, 0.36), PALETTE["certified"])
        arrow(ax, (0.58, 0.18), (0.72, 0.34), PALETTE["ambiguous"])

    flow_panel(
        axes[0],
        "Tox21",
        tox,
        "benchmark_fiber_majority_certify_all",
        "response_certified_exact_fiber",
        "response_certified_then_acquire_ambiguous",
        f"{pct(tox.loc['benchmark_fiber_majority_certify_all', 'mean_error_rate_all_candidates']):.2f}% false",
    )
    panel_label(axes[0], "a")

    flow_panel(
        axes[1],
        "JARVIS",
        jar,
        "local_majority_certify_all",
        "response_certified_error_window",
        "response_certified_then_acquire_ambiguous",
        f"{pct(jar.loc['local_majority_certify_all', 'mean_error_rate_all_candidates']):.1f}% false",
    )
    panel_label(axes[1], "b")

    ax = axes[2]
    for domain, color in [("Tox21", PALETTE["ambiguous"]), ("JARVIS", PALETTE["benchmark"])]:
        sub = cost_curve[cost_curve["domain"] == domain]
        mean = sub.groupby("acquisition_cost_per_candidate")[["baseline_total_cost", "certify_acquire_total_cost"]].mean()
        x = pct(mean.index.to_numpy())
        ax.plot(x, mean["baseline_total_cost"], color=color, ls=(0, (2, 2)), lw=1.0, label=f"{domain}: benchmark action")
        ax.plot(x, mean["certify_acquire_total_cost"], color=color, lw=1.5, label=f"{domain}: certify + acquire")
        be = pct(break_even.loc[domain, "mean_break_even_acquisition_cost"])
        ax.axvline(be, color=color, lw=0.8, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel(r"acquisition cost $\lambda$ (% of one false decision)")
    ax.set_ylabel("expected cost per split")
    ax.set_title("Cost depends on acquisition price", loc="left", pad=2)
    ax.text(1.18, 0.93, "Tox21\n1.18%", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=5.6, color=PALETTE["ambiguous"])
    ax.text(20.7, 0.93, "JARVIS\n20.7%", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=5.6, color=PALETTE["benchmark"])
    ax.legend(frameon=False, loc="upper left", handlelength=1.4, fontsize=5.4, ncol=1)
    clean(ax)
    panel_label(ax, "c")

    ax = axes[3]
    model_counts = choices["best_certification_model"].value_counts()
    categories = ["alignn_model", *model_counts.index.to_list()]
    ypos = {name: i for i, name in enumerate(categories[::-1])}
    rng = np.random.default_rng(11)
    for _, row in choices.iterrows():
        y0 = ypos[row["best_benchmark_model"]] + rng.uniform(-0.07, 0.07)
        y1 = ypos[row["best_certification_model"]] + rng.uniform(-0.07, 0.07)
        ax.plot([0, 1], [y0, y1], color=PALETTE["gray"], alpha=0.30, lw=0.55, zorder=1)
    ax.scatter(
        np.zeros(len(choices)),
        [ypos[m] for m in choices["best_benchmark_model"]],
        color=PALETTE["false"],
        s=28,
        label="benchmark-MAE selected",
        zorder=3,
    )
    for model, count in model_counts.items():
        ax.scatter(1, ypos[model], s=20 + count * 2.4, color=PALETTE["certified"], zorder=3)
        ax.text(1.04, ypos[model], f"{count}", va="center", ha="left", fontsize=5.5, color=PALETTE["certified"])
    model_labels = {
        "alignn_model": "ALIGNN",
        "matminer_rf": "matminer RF",
        "matminer_xgboost": "matminer XGBoost",
        "cfid_chem": "CFID-Chem",
        "cfid": "CFID",
        "kgcnn_coNGN": "kgcnn coNGN",
        "atomgpt_model": "AtomGPT",
        "kgcnn_dimenetPP": "kgcnn DimeNet++",
        "kgcnn_schnet": "kgcnn SchNet",
    }
    ax.set_xlim(-0.12, 1.22)
    ax.set_xticks([0, 1], ["benchmark-MAE\nselected", "certification\nselected"])
    ax.set_yticks([ypos[c] for c in categories], [model_labels.get(c, c.replace("_", " ")) for c in categories])
    ax.set_ylabel("selected model")
    ax.set_title("Model choice changes across splits", loc="left", pad=2)
    ax.text(
        0.03,
        0.08,
        f"{choices['model_choice_changed'].sum():.0f}/50 splits changed model\ncase split: 613 false vs 0",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=1.2),
    )
    clean(ax, ygrid=False)
    panel_label(ax, "d")

    save(fig, "operational_decision_replay")


def active_acquisition() -> None:
    data = pd.read_csv(OUT / "baseline_comparison_acquisition.csv")
    data = data[data["added_probes"] <= 4].copy()
    radius = pd.read_csv(OUT / "baseline_comparison_radius_sensitivity.csv")
    tox = pd.read_csv(OUT / "tox21_probe_acquisition_summary.csv")
    tox_trace = pd.read_csv(OUT / "tox21_probe_acquisition_trace.csv")
    jarvis = pd.read_csv(OUT / "jarvis_probe_acquisition_summary.csv")
    geom = pd.read_csv(OUT / "jarvis_probe_geometry.csv")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(183 * MM, 118 * MM),
        constrained_layout=True,
    )
    axes = axes.ravel()

    ax = axes[0]
    x = data["added_probes"].to_numpy()
    ax.fill_between(
        x,
        pct(data["random_p10_certified"] / data["total_candidates"]),
        pct(data["random_p90_certified"] / data["total_candidates"]),
        color=PALETTE["light_gray"],
        label="random 10-90%",
        zorder=0,
    )
    lines = [
        ("uncertainty_frac", "uncertainty", PALETTE["benchmark"], "s"),
        ("diversity_frac", "diversity", PALETTE["gray"], "^"),
        ("benchmark_aligned_frac", "benchmark-aligned", "#6F6F6F", "v"),
        ("random_mean_frac", "random mean", "#A0A0A0", "o"),
        ("residual_greedy_frac", "residual-greedy", PALETTE["certified"], "o"),
    ]
    for col, label, color, marker in lines:
        ax.plot(x, pct(data[col]), color=color, marker=marker, lw=1.7 if col == "residual_greedy_frac" else 1.0, label=label)
    ax.set_xlabel("added response probes")
    ax.set_ylabel("candidates certified (%)")
    ax.set_xlim(-0.1, 4.1)
    ax.set_ylim(40, 103)
    ax.set_title("Controlled response-space acquisition", loc="left", pad=2)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.55, -0.22),
        ncol=3,
        handlelength=1.0,
        columnspacing=0.9,
    )
    clean(ax)
    panel_label(ax, "a")

    ax = axes[1]
    methods = ["no_new_probe_frac", "random_mean_frac", "benchmark_aligned_frac", "residual_greedy_frac"]
    labels = ["none", "random", "benchmark-aligned", "response-rank"]
    mat = pct(radius[methods].to_numpy())
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=90)
    ax.set_yticks(np.arange(len(radius)), radius["radius_scale"])
    ax.set_xticks(np.arange(len(labels)), labels, rotation=15, ha="right")
    ax.set_xlabel("policy after three probes")
    ax.set_ylabel("radius scale")
    ax.set_title("Radius sensitivity", loc="left", pad=2)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=5.2, color="white" if mat[i, j] > 55 else PALETTE["dark"])
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("certified (%)")
    panel_label(ax, "b")

    ax = axes[2]
    tox_sub = tox[(tox["step"] <= 2) & (tox["policy"].isin(["response_rank", "uncertainty", "random", "benchmark_aligned", "diversity"]))]
    for policy, color in [
        ("response_rank", PALETTE["certified"]),
        ("uncertainty", PALETTE["benchmark"]),
        ("random", PALETTE["gray"]),
        ("benchmark_aligned", "#6F6F6F"),
        ("diversity", PALETTE["ambiguous"]),
    ]:
        sub = tox_sub[tox_sub["policy"] == policy]
        ax.plot(sub["step"], pct(sub["mean_certified_fraction"]), marker="o", color=color, label=policy.replace("_", "-"),
                lw=1.6 if policy == "response_rank" else 0.95)
    # Show the first selected probes for response rank as a compact frequency inset.
    freq = (
        tox_trace[(tox_trace["policy"] == "response_rank") & (tox_trace["step"].isin([1, 2]))]
        .groupby(["step", "last_added_probe"])
        .size()
        .groupby(level=0)
        .apply(lambda s: s / s.sum() * 100)
    )
    ax.text(0.02, 0.95, "response-rank probes:\nSR-MMP then SR-HSE", transform=ax.transAxes, va="top", fontsize=5.8)
    ax.set_xlabel("added assays")
    ax.set_ylabel("certified SR-p53 candidates (%)")
    ax.set_xticks([0, 1, 2])
    ax.set_title("Tox21 held-out probe acquisition", loc="left", pad=2)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.58, -0.22),
        ncol=3,
        handlelength=1.0,
        columnspacing=0.9,
    )
    clean(ax)
    panel_label(ax, "c")

    ax = axes[3]
    jsub = jarvis.set_index("policy").loc[["response_rank", "oracle", "random", "benchmark_aligned"]]
    x = np.arange(len(jsub))
    ax.bar(x - 0.17, pct(jsub["cert_1probe_mean"]), width=0.34, color=PALETTE["certified"], label="certified")
    ax.bar(x + 0.17, pct(jsub["false_dec_1probe_mean"]), width=0.34, color=PALETTE["false"], label="false")
    ax.set_xticks(x, ["response-rank", "oracle", "random", "benchmark-aligned"], rotation=15, ha="right")
    ax.set_ylabel("materials after one probe (%)")
    ax.set_title("JARVIS probe choice", loc="left", pad=2)
    mbj = geom[geom["probe"] == "mbj_bandgap"].iloc[0]
    ax.set_ylim(0, 90)
    label_box = dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.0)
    ax.text(
        -0.30,
        88.0,
        f"MBJ: r={mbj['r_probe_bg']:.2f}, residual reduction {mbj['reduction_pct']:.0f}%",
        ha="left",
        va="top",
        fontsize=5.5,
        bbox=label_box,
    )
    ax.text(
        -0.30,
        82.5,
        "FE-control: r≈-0.42, residual <1%",
        ha="left",
        va="top",
        fontsize=5.5,
        bbox=label_box,
    )
    ax.legend(frameon=False, loc="upper right", handlelength=1.0)
    clean(ax)
    panel_label(ax, "d")

    save(fig, "baseline_comparison_acquisition")


if __name__ == "__main__":
    certification_standard()
    main_results()
    public_audits()
    operational_replay()
    active_acquisition()
