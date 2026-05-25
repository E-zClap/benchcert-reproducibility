"""Nature Computational Science style response-rank demonstrations.

The demonstrations are physically structured synthetic studies: they use
interpretable linear response probes for several scientific domains, but the
candidates are generated examples rather than empirical benchmark records. The
goal is to show how the response-certification diagnostic is computed and
audited.
"""

from __future__ import annotations

import csv
import io
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class Domain:
    name: str
    probes_bench: np.ndarray
    probe_deploy: np.ndarray
    threshold: float
    target: str


def row_span_projector(rows: np.ndarray) -> np.ndarray:
    """Orthogonal projector onto the row span of rows."""
    if rows.size == 0:
        raise ValueError("rows must be non-empty")
    q, _ = np.linalg.qr(rows.T)
    rank = np.linalg.matrix_rank(rows)
    q = q[:, :rank]
    return q @ q.T


def residual_norm(rows: np.ndarray, probe: np.ndarray) -> float:
    p = row_span_projector(rows)
    return float(np.linalg.norm(probe - p @ probe))


def certify(estimate: float, radius: float, threshold: float) -> tuple[str, float, float]:
    lo = estimate - radius
    hi = estimate + radius
    if hi <= threshold:
        label = "certified viable"
    elif lo > threshold:
        label = "certified nonviable"
    else:
        label = "response-ambiguous"
    return label, lo, hi


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_domains() -> list[Domain]:
    # Coordinates are interpretable response directions, not empirical features.
    # Catalysis: [baseline barrier, activation slope, pressure response, poisoning]
    catalysis = Domain(
        name="catalysis",
        probes_bench=np.array(
            [
                [1.0, 0.82, 0.15, 0.00],
                [1.0, 0.76, 0.25, 0.00],
            ]
        ),
        probe_deploy=np.array([1.0, 0.61, 0.65, 0.85]),
        threshold=1.05,
        target="deployment activity cost",
    )

    # Batteries: [baseline fade, C-rate stress, temperature stress, protocol interaction]
    batteries = Domain(
        name="batteries",
        probes_bench=np.array(
            [
                [1.0, 0.30, 0.20, 0.06],
                [1.0, 0.45, 0.25, 0.11],
            ]
        ),
        probe_deploy=np.array([1.0, 0.95, 0.85, 0.81]),
        threshold=1.20,
        target="fast-charge degradation cost",
    )

    # Photonics: [bulk loss, dispersion, roughness sensitivity, interface coupling]
    photonics = Domain(
        name="photonics",
        probes_bench=np.array(
            [
                [1.0, -0.40, 0.00, 0.00],
                [1.0, 0.35, 0.00, 0.00],
            ]
        ),
        probe_deploy=np.array([1.0, 0.15, 0.70, 0.90]),
        threshold=1.00,
        target="integrated-device optical loss",
    )

    # Quantum-materials spin-defect proxy: [host, bath, optical, charge]
    spin_defects = Domain(
        name="spin defects",
        probes_bench=np.array(
            [
                [1.0, 0.20, 0.05, 0.00],
                [0.9, 0.25, 0.10, 0.00],
            ]
        ),
        probe_deploy=np.array([0.95, 0.75, 0.55, 0.60]),
        threshold=1.10,
        target="qubit viability cost",
    )
    return [catalysis, batteries, photonics, spin_defects]


def candidate_rows(domain: Domain, rng: np.random.Generator, n: int = 24) -> list[dict[str, object]]:
    resid = residual_norm(domain.probes_bench, domain.probe_deploy)
    rows: list[dict[str, object]] = []
    for i in range(n):
        # Candidate central deployment estimates are spread around threshold.
        estimate = float(domain.threshold + rng.normal(0.0, 0.22))
        # Benchmark error is low for most candidates to mimic benchmark-success filtering.
        benchmark_error = float(abs(rng.normal(0.025, 0.015)))
        # Uncertainty radius differs by candidate descriptor completeness.
        descriptor_radius = float(rng.uniform(0.05, 0.65))
        ambiguity_radius = descriptor_radius * resid
        label, lo, hi = certify(estimate, ambiguity_radius, domain.threshold)
        # AI score is deliberately tied to central estimate and benchmark error, not residual.
        score = float(sigmoid(np.array([(domain.threshold - estimate) * 5.0 - benchmark_error * 8.0]))[0])
        rows.append(
            {
                "domain": domain.name,
                "candidate": f"{domain.name[:3].upper()}-{i + 1:02d}",
                "ai_score": score,
                "benchmark_error": benchmark_error,
                "deployment_estimate": estimate,
                "threshold": domain.threshold,
                "residual_norm": resid,
                "descriptor_radius": descriptor_radius,
                "ambiguity_radius": ambiguity_radius,
                "interval_low": lo,
                "interval_high": hi,
                "certification_class": label,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_certification_outputs(domains: list[Domain]) -> list[dict[str, object]]:
    rng = np.random.default_rng(7)
    rows: list[dict[str, object]] = []
    for domain in domains:
        rows.extend(candidate_rows(domain, rng))

    write_csv(OUTDIR / "ncs_certification_table.csv", rows)

    summary_rows: list[dict[str, object]] = []
    for domain in domains:
        domain_rows = [r for r in rows if r["domain"] == domain.name]
        counts = Counter(r["certification_class"] for r in domain_rows)
        summary_rows.append(
            {
                "domain": domain.name,
                "deployment_target": domain.target,
                "residual_norm": residual_norm(domain.probes_bench, domain.probe_deploy),
                "certified_viable": counts["certified viable"],
                "certified_nonviable": counts["certified nonviable"],
                "response_ambiguous": counts["response-ambiguous"],
            }
        )
    write_csv(OUTDIR / "ncs_domain_summary.csv", summary_rows)

    # Bar chart by class.
    fig, ax = plt.subplots(figsize=(9.5, 4.3), constrained_layout=True)
    names = [d.name for d in domains]
    viable = [r["certified_viable"] for r in summary_rows]
    nonviable = [r["certified_nonviable"] for r in summary_rows]
    ambiguous = [r["response_ambiguous"] for r in summary_rows]
    x = np.arange(len(names))
    ax.bar(x, viable, label="certified viable", color="#188a5b")
    ax.bar(x, nonviable, bottom=viable, label="certified nonviable", color="#6f7f95")
    ax.bar(
        x,
        ambiguous,
        bottom=np.array(viable) + np.array(nonviable),
        label="response-ambiguous",
        color="#c43b3b",
    )
    ax.set_xticks(x, names, rotation=12, ha="right")
    ax.set_ylabel("candidate count")
    ax.set_title("Response-rank certification across synthetic scientific domains")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.savefig(OUTDIR / "ncs_certification_by_domain.svg")
    fig.savefig(OUTDIR / "ncs_certification_by_domain.png", dpi=200)

    # Scatter AI score vs ambiguity radius.
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    colors = {
        "certified viable": "#188a5b",
        "certified nonviable": "#6f7f95",
        "response-ambiguous": "#c43b3b",
    }
    for label, color in colors.items():
        selected = [r for r in rows if r["certification_class"] == label]
        ax.scatter(
            [r["ai_score"] for r in selected],
            [r["ambiguity_radius"] for r in selected],
            s=38,
            alpha=0.82,
            label=label,
            color=color,
        )
    ax.set_xlabel("benchmark-style AI score")
    ax.set_ylabel("response-certification gap")
    ax.set_title("High AI score does not imply low deployment ambiguity")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUTDIR / "ncs_score_vs_gap.svg")
    fig.savefig(OUTDIR / "ncs_score_vs_gap.png", dpi=200)
    return rows


def classify_many(
    estimates: np.ndarray,
    radii: np.ndarray,
    residuals: np.ndarray,
    threshold: float,
) -> int:
    """Count candidates ambiguous under at least one deployment probe."""
    ambiguous = 0
    for est, r in zip(estimates, radii):
        width = float(np.max(r * residuals))
        lo = est - width
        hi = est + width
        if lo <= threshold < hi:
            ambiguous += 1
    return ambiguous


def active_probe_experiment() -> None:
    rng = np.random.default_rng(11)
    dim = 8
    n_candidates = 120
    threshold = 1.0
    estimates = threshold + rng.normal(0.0, 0.18, size=n_candidates)
    radii = rng.uniform(0.08, 0.55, size=n_candidates)

    initial = rng.normal(size=(2, dim))
    deploy = rng.normal(size=(12, dim))
    # Candidate probes are possible new measurements/simulations.
    pool = rng.normal(size=(20, dim))

    def residuals_for(rows: np.ndarray) -> np.ndarray:
        p = row_span_projector(rows)
        return np.linalg.norm(deploy - deploy @ p, axis=1)

    active_rows = initial.copy()
    active_counts = []
    selected = []
    for step in range(7):
        residuals = residuals_for(active_rows)
        active_counts.append(classify_many(estimates, radii, residuals, threshold))
        if step == 6:
            break
        best_idx = None
        best_score = np.inf
        for j, probe in enumerate(pool):
            if j in selected:
                continue
            trial = np.vstack([active_rows, probe])
            score = float(np.max(residuals_for(trial)))
            if score < best_score:
                best_score = score
                best_idx = j
        assert best_idx is not None
        selected.append(best_idx)
        active_rows = np.vstack([active_rows, pool[best_idx]])

    random_trials = []
    for trial_seed in range(80):
        local = np.random.default_rng(1000 + trial_seed)
        order = local.permutation(len(pool))
        rows = initial.copy()
        counts = []
        for step in range(7):
            residuals = residuals_for(rows)
            counts.append(classify_many(estimates, radii, residuals, threshold))
            if step < 6:
                rows = np.vstack([rows, pool[order[step]]])
        random_trials.append(counts)
    random_trials = np.array(random_trials)
    random_mean = random_trials.mean(axis=0)
    random_low = np.percentile(random_trials, 10, axis=0)
    random_high = np.percentile(random_trials, 90, axis=0)

    # A benchmark-oriented baseline: choose probes that are most aligned with
    # the current benchmark span. This mimics improving coverage near the
    # benchmark distribution rather than targeting deployment residuals.
    benchmark_rows = initial.copy()
    benchmark_counts = []
    benchmark_selected = []
    for step in range(7):
        residuals = residuals_for(benchmark_rows)
        benchmark_counts.append(classify_many(estimates, radii, residuals, threshold))
        if step == 6:
            break
        p = row_span_projector(benchmark_rows)
        best_idx = None
        best_score = -np.inf
        for j, probe in enumerate(pool):
            if j in benchmark_selected:
                continue
            score = float(np.linalg.norm(p @ probe) / (np.linalg.norm(probe) + 1e-12))
            if score > best_score:
                best_score = score
                best_idx = j
        assert best_idx is not None
        benchmark_selected.append(best_idx)
        benchmark_rows = np.vstack([benchmark_rows, pool[best_idx]])

    rows_out = []
    for step in range(7):
        rows_out.append(
            {
                "added_probes": step,
                "active_ambiguous": int(active_counts[step]),
                "benchmark_oriented_ambiguous": int(benchmark_counts[step]),
                "random_mean_ambiguous": float(random_mean[step]),
                "random_p10_ambiguous": float(random_low[step]),
                "random_p90_ambiguous": float(random_high[step]),
            }
        )
    write_csv(OUTDIR / "ncs_active_probe_experiment.csv", rows_out)

    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    x = np.arange(7)
    ax.plot(x, active_counts, marker="o", color="#188a5b", label="residual-greedy probes")
    ax.plot(
        x,
        benchmark_counts,
        marker="^",
        color="#b16b00",
        label="benchmark-oriented probes",
    )
    ax.plot(x, random_mean, marker="s", color="#6f7f95", label="random probes (mean)")
    ax.fill_between(x, random_low, random_high, color="#6f7f95", alpha=0.18, label="random 10-90%")
    ax.set_xlabel("added probes")
    ax.set_ylabel("response-ambiguous candidates")
    ax.set_title("Active probe selection reduces response ambiguity")
    ax.legend(frameon=False)
    fig.savefig(OUTDIR / "ncs_active_probe_experiment.svg")
    fig.savefig(OUTDIR / "ncs_active_probe_experiment.png", dpi=200)


def public_spin_defect_dataset_audit() -> None:
    """Audit host-only certification on a public 2D spin-coherence dataset.

    Data source: Toriyama et al., Dataset: "Strategies to search for
    two-dimensional materials with long spin qubit coherence time",
    Zenodo record 16996230. The audit uses the reported 2D-host T2 values and
    heterostructure T2 values. It does not recompute CCE simulations.
    """

    import pandas as pd

    base = "https://zenodo.org/api/records/16996230/files"
    material_url = f"{base}/Data_2D-Materials.xlsx/content"
    hetero_url = f"{base}/Data_Heterostructures.xlsx/content"

    with urlopen(material_url, timeout=30) as response:
        materials = pd.read_excel(io.BytesIO(response.read()))
    with urlopen(hetero_url, timeout=30) as response:
        hetero = pd.read_excel(io.BytesIO(response.read()))

    hetero = hetero.dropna(subset=["T2"])
    merged = hetero.merge(
        materials[["MC2D_UUID", "T2"]],
        left_on="Host_2D_MC2D_UUID",
        right_on="MC2D_UUID",
        suffixes=("_heterostructure", "_bare"),
    )

    threshold_ms = 1.0
    grouped = (
        merged.groupby(["Compound", "Host_2D_MC2D_UUID"])
        .agg(
            bare_t2_ms=("T2_bare", "first"),
            min_heterostructure_t2_ms=("T2_heterostructure", "min"),
            max_heterostructure_t2_ms=("T2_heterostructure", "max"),
            mean_heterostructure_t2_ms=("T2_heterostructure", "mean"),
            n_substrates=("Substrate", "nunique"),
        )
        .reset_index()
    )

    classes = []
    for _, row in grouped.iterrows():
        if row["min_heterostructure_t2_ms"] >= threshold_ms:
            classes.append("certified viable")
        elif row["max_heterostructure_t2_ms"] < threshold_ms:
            classes.append("certified nonviable")
        else:
            classes.append("response-ambiguous")
    grouped["certification_class"] = classes
    grouped["host_only_gap_ms"] = (
        grouped["max_heterostructure_t2_ms"] - grouped["min_heterostructure_t2_ms"]
    )
    grouped.to_csv(OUTDIR / "ncs_public_spin_defect_audit.csv", index=False)

    counts = grouped["certification_class"].value_counts()
    summary = [
        {
            "threshold_ms": threshold_ms,
            "hosts_with_valid_heterostructure_t2": int(len(grouped)),
            "heterostructures_with_valid_t2": int(len(merged)),
            "certified_viable": int(counts.get("certified viable", 0)),
            "certified_nonviable": int(counts.get("certified nonviable", 0)),
            "response_ambiguous": int(counts.get("response-ambiguous", 0)),
        }
    ]
    write_csv(OUTDIR / "ncs_public_spin_defect_summary.csv", summary)

    colors = {
        "certified viable": "#188a5b",
        "certified nonviable": "#6f7f95",
        "response-ambiguous": "#c43b3b",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)

    ax = axes[0]
    order = ["certified viable", "response-ambiguous", "certified nonviable"]
    ax.bar(
        order,
        [int(counts.get(label, 0)) for label in order],
        color=[colors[label] for label in order],
    )
    ax.set_ylabel("2D host count")
    ax.set_title("Host-only deployment certification")
    ax.tick_params(axis="x", rotation=15)

    ax = axes[1]
    for label in order:
        selected = grouped[grouped["certification_class"] == label]
        ax.scatter(
            selected["bare_t2_ms"],
            selected["min_heterostructure_t2_ms"],
            s=28,
            alpha=0.82,
            color=colors[label],
            label=label,
        )
    ax.axhline(threshold_ms, color="#202736", linestyle="--", linewidth=1.2)
    ax.axvline(threshold_ms, color="#202736", linestyle=":", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("bare 2D host T2 (ms)")
    ax.set_ylabel("minimum heterostructure T2 across substrates (ms)")
    ax.set_title("Substrate deployment can change certification")
    ax.legend(frameon=False, fontsize=8)

    fig.savefig(OUTDIR / "ncs_public_spin_defect_audit.svg")
    fig.savefig(OUTDIR / "ncs_public_spin_defect_audit.png", dpi=200)


def main() -> None:
    domains = make_domains()
    make_certification_outputs(domains)
    active_probe_experiment()
    public_spin_defect_dataset_audit()


if __name__ == "__main__":
    main()
