"""Shadow-prospective response-probe acquisition on Tox21.

This is the closed-loop experiment that matches the response-rank theory more
closely than candidate-label acquisition.  The policy does not buy one
deployment label for one compound.  It chooses which response channel/assay to
add next, then certifies held-out SR-p53 decisions from the expanded response
panel.

Protocol
--------
* Deployment response: SR-p53.
* Initial benchmark response panel: the seven nuclear-receptor assays.
* Candidate response probes: the four non-p53 stress-response assays.
* Calibration compounds are used to choose probes and build finite-fiber
  certificates.
* Held-out compounds are scored only after the probe order is frozen.

The main response-rank policy chooses the next assay that maximizes the
calibration-supported certified fraction after adding that assay.  This is the
finite-fiber analogue of choosing the probe that most reduces the remaining
deployment ambiguity.  Baselines choose probes by assay uncertainty, diversity,
benchmark alignment or random order.

Outputs
-------
outputs/tox21_probe_acquisition_trace.csv
outputs/tox21_probe_acquisition_summary.csv
outputs/tox21_probe_acquisition.pdf/.png/.svg
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(exist_ok=True)

TOX21_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
DEPLOYMENT = "SR-p53"
INITIAL_PROBES = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
]
POOL_PROBES = ["SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP"]
POLICIES = [
    "response_rank",
    "uncertainty",
    "diversity",
    "benchmark_aligned",
    "random",
    "oracle",
]


def fetch_tox21() -> pd.DataFrame:
    request = Request(TOX21_URL, headers={"User-Agent": "Codex"})
    with urlopen(request, timeout=90) as response:
        data = response.read()
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
        return pd.read_csv(gz)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pattern_frame(df: pd.DataFrame, probes: list[str]) -> pd.Series:
    if not probes:
        return pd.Series([""] * len(df), index=df.index)
    return df[probes].astype(int).astype(str).agg("|".join, axis=1)


def certificate_map(cal: pd.DataFrame, probes: list[str], min_support: int) -> dict[str, int | None]:
    patterns = pattern_frame(cal, probes)
    mapping: dict[str, int | None] = {}
    for pat, group in cal.groupby(patterns):
        labels = group[DEPLOYMENT].astype(int).to_numpy()
        if len(labels) < min_support:
            mapping[str(pat)] = None
        elif np.all(labels == 0):
            mapping[str(pat)] = 0
        elif np.all(labels == 1):
            mapping[str(pat)] = 1
        else:
            mapping[str(pat)] = None
    return mapping


def apply_certificate(
    cal: pd.DataFrame,
    test: pd.DataFrame,
    probes: list[str],
    min_support: int,
) -> np.ndarray:
    mapping = certificate_map(cal, probes, min_support)
    test_patterns = pattern_frame(test, probes)
    decisions = np.full(len(test), -1, dtype=int)
    for i, pat in enumerate(test_patterns):
        decision = mapping.get(str(pat), None)
        if decision is not None:
            decisions[i] = int(decision)
    return decisions


def evaluate(cal: pd.DataFrame, test: pd.DataFrame, probes: list[str], min_support: int) -> dict[str, float]:
    decisions = apply_certificate(cal, test, probes, min_support)
    y = test[DEPLOYMENT].astype(int).to_numpy()
    certified = decisions >= 0
    false = certified & (decisions != y)
    active_found = certified & (decisions == 1) & (y == 1)
    inactive_found = certified & (decisions == 0) & (y == 0)
    n_cert = int(np.sum(certified))
    return {
        "n_certified": float(n_cert),
        "certified_fraction": float(n_cert / len(test)),
        "false_decisions": float(np.sum(false)),
        "false_decision_rate_all": float(np.sum(false) / len(test)),
        "error_rate_among_certified": float(np.sum(false) / n_cert) if n_cert else np.nan,
        "active_found": float(np.sum(active_found)),
        "inactive_found": float(np.sum(inactive_found)),
        "active_yield": float(np.sum(active_found) / max(1, int(np.sum(y == 1)))),
        "inactive_yield": float(np.sum(inactive_found) / max(1, int(np.sum(y == 0)))),
    }


def calibration_score(cal: pd.DataFrame, probes: list[str], min_support: int) -> float:
    # Leave-in score used only to choose a response probe.  A candidate probe is
    # useful when it makes more calibration compounds fall into supported pure
    # SR-p53 fibers.
    decisions = apply_certificate(cal, cal, probes, min_support)
    return float(np.mean(decisions >= 0))


def label_entropy(x: np.ndarray) -> float:
    p = float(np.mean(x))
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))


def choose_probe(
    policy: str,
    cal: pd.DataFrame,
    test: pd.DataFrame,
    selected: list[str],
    remaining: list[str],
    min_support: int,
    rng: np.random.Generator,
) -> str:
    if policy == "random":
        return str(rng.choice(remaining))

    if policy == "response_rank":
        # Finite-fiber residual-greedy analogue: choose the response probe that
        # most increases supported pure deployment fibers on calibration data.
        return max(
            remaining,
            key=lambda q: (
                calibration_score(cal, selected + [q], min_support),
                -len(selected + [q]),
                q,
            ),
        )

    if policy == "uncertainty":
        # Generic assay-uncertainty baseline: choose the assay whose own binary
        # response is closest to balanced among calibration compounds.
        return max(remaining, key=lambda q: (label_entropy(cal[q].astype(int).to_numpy()), q))

    if policy == "diversity":
        # Choose the assay least correlated with the already measured panel.
        x_sel = cal[selected].astype(float).to_numpy()
        scores = {}
        for q in remaining:
            xq = cal[q].astype(float).to_numpy()
            corrs = []
            for j in range(x_sel.shape[1]):
                if np.std(x_sel[:, j]) < 1e-12 or np.std(xq) < 1e-12:
                    corrs.append(0.0)
                else:
                    corrs.append(abs(float(np.corrcoef(x_sel[:, j], xq)[0, 1])))
            scores[q] = float(np.mean(corrs))
        return min(remaining, key=lambda q: (scores[q], q))

    if policy == "benchmark_aligned":
        # Choose the assay most redundant with the existing benchmark panel.
        x_sel = cal[selected].astype(float).to_numpy()
        scores = {}
        for q in remaining:
            xq = cal[q].astype(float).to_numpy()
            corrs = []
            for j in range(x_sel.shape[1]):
                if np.std(x_sel[:, j]) < 1e-12 or np.std(xq) < 1e-12:
                    corrs.append(0.0)
                else:
                    corrs.append(abs(float(np.corrcoef(x_sel[:, j], xq)[0, 1])))
            scores[q] = float(np.mean(corrs))
        return max(remaining, key=lambda q: (scores[q], q))

    if policy == "oracle":
        # Upper bound: choose the probe that would certify most held-out
        # compounds.  This uses test deployment labels and is not deployable.
        return max(
            remaining,
            key=lambda q: (
                evaluate(cal, test, selected + [q], min_support)["certified_fraction"],
                q,
            ),
        )

    raise ValueError(policy)


def run_campaign(
    data: pd.DataFrame,
    seed: int,
    min_support: int,
    calibration_fraction: float,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(31000 + seed)
    idx = rng.permutation(len(data))
    n_cal = int(round(calibration_fraction * len(data)))
    cal = data.iloc[idx[:n_cal]].reset_index(drop=True)
    test = data.iloc[idx[n_cal:]].reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for policy in POLICIES:
        selected = list(INITIAL_PROBES)
        remaining = list(POOL_PROBES)
        policy_rng = np.random.default_rng(41000 + seed * 101 + POLICIES.index(policy))
        for step in range(len(POOL_PROBES) + 1):
            metrics = evaluate(cal, test, selected, min_support)
            rows.append(
                {
                    "seed": seed,
                    "policy": policy,
                    "step": step,
                    "n_selected_probes": len(selected),
                    "selected_probes": ";".join(selected),
                    "last_added_probe": selected[-1] if step else "",
                    "n_calibration": len(cal),
                    "n_test": len(test),
                    "deployment_active_prevalence": float(test[DEPLOYMENT].mean()),
                    **metrics,
                }
            )
            if not remaining:
                break
            q = choose_probe(policy, cal, test, selected, remaining, min_support, policy_rng)
            selected.append(q)
            remaining.remove(q)
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for policy in POLICIES:
        for step in sorted({int(r["step"]) for r in rows}):
            sub = [r for r in rows if r["policy"] == policy and int(r["step"]) == step]
            if not sub:
                continue
            added = Counter(str(r["last_added_probe"]) for r in sub if str(r["last_added_probe"]))
            out.append(
                {
                    "policy": policy,
                    "step": step,
                    "n_runs": len(sub),
                    "modal_last_added_probe": added.most_common(1)[0][0] if added else "",
                    "mean_certified_fraction": float(np.mean([float(r["certified_fraction"]) for r in sub])),
                    "mean_false_decision_rate_all": float(
                        np.mean([float(r["false_decision_rate_all"]) for r in sub])
                    ),
                    "mean_error_rate_among_certified": float(
                        np.nanmean([float(r["error_rate_among_certified"]) for r in sub])
                    ),
                    "mean_active_yield": float(np.mean([float(r["active_yield"]) for r in sub])),
                    "mean_inactive_yield": float(np.mean([float(r["inactive_yield"]) for r in sub])),
                }
            )
    return out


def plot(rows: list[dict[str, object]]) -> None:
    colors = {
        "response_rank": "#188a5b",
        "uncertainty": "#4f7cac",
        "diversity": "#8b6f47",
        "benchmark_aligned": "#d46a3a",
        "random": "#6f7f95",
        "oracle": "#111111",
    }
    labels = {
        "response_rank": "response-rank",
        "uncertainty": "uncertainty",
        "diversity": "diversity",
        "benchmark_aligned": "benchmark-aligned",
        "random": "random",
        "oracle": "oracle",
    }
    steps = sorted({int(r["step"]) for r in rows})
    plt.rcParams.update({
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.2), constrained_layout=True)
    specs = [
        ("certified_fraction", "Certified candidates (%)", 100.0),
        ("false_decision_rate_all", "False decisions (%)", 100.0),
        ("inactive_yield", "Certified inactive yield (%)", 100.0),
    ]
    for ax, (metric, ylabel, scale) in zip(axes, specs):
        for policy in POLICIES:
            vals = []
            for step in steps:
                sub = [float(r[metric]) for r in rows if r["policy"] == policy and int(r["step"]) == step]
                vals.append(float(np.nanmean(sub)) * scale)
            ax.plot(steps, vals, marker="o", lw=1.6, ms=3, color=colors[policy], label=labels[policy])
        ax.set_xlabel("added response probes")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#dddddd", lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="best")
    fig.savefig(OUTDIR / "tox21_probe_acquisition.pdf")
    fig.savefig(OUTDIR / "tox21_probe_acquisition.svg")
    fig.savefig(OUTDIR / "tox21_probe_acquisition.png", dpi=600)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=200)
    parser.add_argument("--min-support", type=int, default=50)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    args = parser.parse_args()

    df = fetch_tox21()
    cols = INITIAL_PROBES + POOL_PROBES + [DEPLOYMENT]
    data = df[cols].dropna().copy()
    data[cols] = data[cols].astype(int)

    rows: list[dict[str, object]] = []
    for seed in range(args.n_seeds):
        rows.extend(run_campaign(data, seed, args.min_support, args.calibration_fraction))
    write_csv(OUTDIR / "tox21_probe_acquisition_trace.csv", rows)
    summary = summarize(rows)
    write_csv(OUTDIR / "tox21_probe_acquisition_summary.csv", summary)
    plot(rows)
    early = [r for r in summary if int(r["step"]) in {1, 2}]
    final = [r for r in summary if int(r["step"]) == len(POOL_PROBES)]
    print(f"Ran {args.n_seeds} response-probe acquisition campaigns.")
    print("Early-budget summary (the relevant comparison):")
    for row in early:
        print(row)
    print("Final summary after all probes are acquired:")
    for row in final:
        print(row)


if __name__ == "__main__":
    main()
