"""Synthetic deployment-decision sufficiency experiment.

Response-rank certification is the linear benchmark-response-only case of a
more general rule: certify a deployment action only when that action is
invariant over every admissible system compatible with the declared evidence.

This experiment shows the gain from valid extra structure.  The benchmark
observes one response coordinate b.  The deployment response also depends on a
hidden coordinate h.  Without structural information, h can vary in a large
ambient interval and response-rank/ambient certification is conservative.  With
a declared nonlinear admissible relation h ~= sin(pi b), the feasible deployment
set is much narrower, so the general decision-sufficiency certificate can
certify many more decisions while preserving zero false certificates.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def certify_interval(low: np.ndarray, high: np.ndarray, tau: float) -> np.ndarray:
    decisions = np.full(len(low), -1, dtype=int)
    decisions[high <= tau] = 0
    decisions[low > tau] = 1
    return decisions


def evaluate(decisions: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    certified = decisions >= 0
    false = certified & (decisions != truth)
    return {
        "certified_fraction": float(np.mean(certified)),
        "ambiguous_fraction": float(1.0 - np.mean(certified)),
        "false_decision_rate_all": float(np.mean(false)),
        "error_rate_among_certified": float(np.sum(false) / np.sum(certified))
        if np.any(certified)
        else np.nan,
    }


def run_one(seed: int, n: int, gamma: float, tau: float, rho: float, ambient_h: float) -> list[dict[str, object]]:
    rng = np.random.default_rng(90000 + seed)
    b = rng.uniform(-1.0, 1.0, n)
    hidden_center = np.sin(np.pi * b)
    h = hidden_center + rng.uniform(-rho, rho, n)
    y = b + gamma * h
    truth = (y > tau).astype(int)

    rows: list[dict[str, object]] = []

    # Benchmark action: act as though the benchmark coordinate alone were the
    # deployment response.  This is intentionally not a certificate.
    benchmark_decision = (b > tau).astype(int)
    rows.append({"seed": seed, "protocol": "benchmark_action", **evaluate(benchmark_decision, truth)})

    # Ambient response-rank/benchmark-only certificate: h is only known to lie
    # in a broad admissible ambient interval.
    ambient_low = b - gamma * ambient_h
    ambient_high = b + gamma * ambient_h
    ambient_decision = certify_interval(ambient_low, ambient_high, tau)
    rows.append({"seed": seed, "protocol": "ambient_response_rank", **evaluate(ambient_decision, truth)})

    # General deployment-decision sufficiency certificate: use the declared
    # nonlinear admissible relation for h.  This is the same invariant-decision
    # rule, but over a smaller evidence-compatible set.
    struct_low = b + gamma * (hidden_center - rho)
    struct_high = b + gamma * (hidden_center + rho)
    struct_decision = certify_interval(struct_low, struct_high, tau)
    rows.append({"seed": seed, "protocol": "decision_sufficiency", **evaluate(struct_decision, truth)})

    # Oracle deployment label included only as an upper bound.
    rows.append({"seed": seed, "protocol": "oracle_deployment", **evaluate(truth, truth)})
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for protocol in ["benchmark_action", "ambient_response_rank", "decision_sufficiency", "oracle_deployment"]:
        sub = [r for r in rows if r["protocol"] == protocol]
        out.append(
            {
                "protocol": protocol,
                "n_runs": len(sub),
                "mean_certified_fraction": float(np.mean([float(r["certified_fraction"]) for r in sub])),
                "mean_ambiguous_fraction": float(np.mean([float(r["ambiguous_fraction"]) for r in sub])),
                "mean_false_decision_rate_all": float(np.mean([float(r["false_decision_rate_all"]) for r in sub])),
                "mean_error_rate_among_certified": float(
                    np.nanmean([float(r["error_rate_among_certified"]) for r in sub])
                ),
            }
        )
    return out


def plot(summary: list[dict[str, object]]) -> None:
    labels = ["benchmark\naction", "ambient\nrank", "decision\nsufficiency", "oracle"]
    protocols = ["benchmark_action", "ambient_response_rank", "decision_sufficiency", "oracle_deployment"]
    lookup = {r["protocol"]: r for r in summary}
    cert = [100 * float(lookup[p]["mean_certified_fraction"]) for p in protocols]
    false = [100 * float(lookup[p]["mean_false_decision_rate_all"]) for p in protocols]

    plt.rcParams.update({
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.1), constrained_layout=True)
    colors = ["#c43b3b", "#4f7cac", "#188a5b", "#6f7f95"]
    axes[0].bar(range(len(cert)), cert, color=colors, width=0.62)
    axes[0].set_xticks(range(len(cert)), labels)
    axes[0].set_ylabel("Certified decisions (%)")
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis="y", color="#dddddd", lw=0.5)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[1].bar(range(len(false)), false, color=colors, width=0.62)
    axes[1].set_xticks(range(len(false)), labels)
    axes[1].set_ylabel("False decisions (%)")
    axes[1].set_ylim(0, max(false + [1.0]) * 1.25)
    axes[1].grid(axis="y", color="#dddddd", lw=0.5)
    axes[1].spines[["top", "right"]].set_visible(False)
    for ax, vals in zip(axes, [cert, false]):
        for i, val in enumerate(vals):
            ax.text(i, val + max(vals + [1.0]) * 0.03, f"{val:.1f}", ha="center", fontsize=6)
    fig.savefig(OUT / "decision_sufficiency_generalization.pdf")
    fig.savefig(OUT / "decision_sufficiency_generalization.svg")
    fig.savefig(OUT / "decision_sufficiency_generalization.png", dpi=600)
    plt.close(fig)


def main() -> None:
    n_runs = 100
    rows: list[dict[str, object]] = []
    for seed in range(n_runs):
        rows.extend(run_one(seed, n=1000, gamma=0.9, tau=0.4, rho=0.12, ambient_h=1.2))
    write_csv(OUT / "decision_sufficiency_generalization_runs.csv", rows)
    summary = summarize(rows)
    write_csv(OUT / "decision_sufficiency_generalization_summary.csv", summary)
    plot(summary)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
