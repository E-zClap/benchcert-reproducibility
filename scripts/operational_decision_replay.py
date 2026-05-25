"""Operational decision replays for response-rank certification.

These experiments convert the finite-fiber audits into held-out decision
procedures:

* Tox21: exact benchmark-assay fibers are learned on a calibration split and
  tested on held-out compounds.
* JARVIS: continuous formation-energy prediction fibers are learned on a
  calibration split and tested on held-out materials for a band-gap threshold.

The goal is not to train new models. It is to ask whether the response-certified
trichotomy changes an operational choice before deployment labels are revealed.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import Request, urlopen
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from revision_robustness_experiments import fetch_bytes


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(exist_ok=True)

TOX21_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
TOX21_BENCHMARK_ASSAYS = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
]
TOX21_DEPLOYMENT_ASSAY = "SR-p53"

JARVIS_RAW = "https://raw.githubusercontent.com/usnistgov/jarvis_leaderboard/main"
JARVIS_TREE_API = (
    "https://api.github.com/repos/usnistgov/jarvis_leaderboard/git/trees/main?recursive=1"
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fetch_tox21() -> pd.DataFrame:
    request = Request(TOX21_URL, headers={"User-Agent": "Codex"})
    with urlopen(request, timeout=90) as response:
        data = response.read()
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
        return pd.read_csv(gz)


def tox21_replay(n_seeds: int = 100, calibration_fraction: float = 0.5, min_support: int = 50) -> None:
    df = fetch_tox21()
    cols = TOX21_BENCHMARK_ASSAYS + [TOX21_DEPLOYMENT_ASSAY]
    data = df[cols].dropna().copy()
    data[cols] = data[cols].astype(int)
    patterns = data[TOX21_BENCHMARK_ASSAYS].astype(str).agg("|".join, axis=1).to_numpy()
    y = data[TOX21_DEPLOYMENT_ASSAY].to_numpy(dtype=int)

    rows: list[dict[str, object]] = []
    n = len(data)
    for seed in range(n_seeds):
        rng = np.random.default_rng(8100 + seed)
        idx = rng.permutation(n)
        n_cal = int(round(calibration_fraction * n))
        cal_idx, test_idx = idx[:n_cal], idx[n_cal:]

        # Calibration-only exact-fiber certification.
        fiber_labels: dict[str, str] = {}
        cal_counts: dict[str, Counter[int]] = defaultdict(Counter)
        for i in cal_idx:
            cal_counts[patterns[i]][int(y[i])] += 1
        for pat, counts in cal_counts.items():
            total = sum(counts.values())
            if total < min_support:
                fiber_labels[pat] = "ambiguous"
            elif counts[1] == 0:
                fiber_labels[pat] = "certified_inactive"
            elif counts[0] == 0:
                fiber_labels[pat] = "certified_active"
            else:
                fiber_labels[pat] = "ambiguous"

        # Majority fiber baseline: always certify the calibration majority label.
        majority_labels: dict[str, int] = {}
        global_majority = int(np.mean(y[cal_idx]) >= 0.5)
        for pat, counts in cal_counts.items():
            majority_labels[pat] = 1 if counts[1] > counts[0] else 0

        def add_row(policy: str, decisions: list[int | None], acquisition_cost: int = 0) -> None:
            certified = [d is not None for d in decisions]
            n_cert = int(sum(certified))
            errors = int(
                sum(
                    int(decisions[j] != int(y[test_idx[j]]))
                    for j in range(len(test_idx))
                    if decisions[j] is not None
                )
            )
            rows.append(
                {
                    "seed": seed,
                    "policy": policy,
                    "n_test": len(test_idx),
                    "n_certified": n_cert,
                    "certified_fraction": n_cert / len(test_idx),
                    "n_ambiguous": len(test_idx) - n_cert,
                    "error_count": errors,
                    "error_rate_among_certified": errors / n_cert if n_cert else math.nan,
                    "error_rate_all_candidates": errors / len(test_idx),
                    "deployment_active_prevalence": float(np.mean(y[test_idx])),
                    "acquisition_cost": acquisition_cost,
                }
            )

        # Policy 1: response-certified exact fibers.
        rc_decisions: list[int | None] = []
        for i in test_idx:
            label = fiber_labels.get(patterns[i], "ambiguous")
            if label == "certified_inactive":
                rc_decisions.append(0)
            elif label == "certified_active":
                rc_decisions.append(1)
            else:
                rc_decisions.append(None)
        add_row("response_certified_exact_fiber", rc_decisions)

        # Policy 2: majority-by-fiber benchmark-only classifier.
        maj_decisions = [majority_labels.get(patterns[i], global_majority) for i in test_idx]
        add_row("benchmark_fiber_majority_certify_all", maj_decisions)

        # Policy 3: naive global deployment prevalence classifier.
        global_decisions = [global_majority for _ in test_idx]
        add_row("global_majority_certify_all", global_decisions)

        # Policy 4: response-certified + acquire deployment assay for ambiguous cases.
        acquired = [int(y[i]) if rc_decisions[j] is None else rc_decisions[j] for j, i in enumerate(test_idx)]
        add_row("response_certified_then_acquire_ambiguous", acquired, acquisition_cost=sum(d is None for d in rc_decisions))

    write_csv(OUTDIR / "tox21_operational_replay.csv", rows)
    summarize_policy_rows(rows, OUTDIR / "tox21_operational_replay_summary.csv")


def summarize_policy_rows(rows: list[dict[str, object]], path: Path) -> None:
    out: list[dict[str, object]] = []
    for policy in sorted({str(row["policy"]) for row in rows}):
        sub = [row for row in rows if row["policy"] == policy]
        out.append(
            {
                "policy": policy,
                "n_runs": len(sub),
                "mean_certified_fraction": float(np.mean([float(r["certified_fraction"]) for r in sub])),
                "mean_error_rate_among_certified": float(
                    np.nanmean([float(r["error_rate_among_certified"]) for r in sub])
                ),
                "mean_error_rate_all_candidates": float(np.mean([float(r["error_rate_all_candidates"]) for r in sub])),
                "mean_ambiguous": float(np.mean([float(r["n_ambiguous"]) for r in sub])),
                "mean_acquisition_cost": float(np.mean([float(r["acquisition_cost"]) for r in sub])),
            }
        )
    write_csv(path, out)


def github_tree_paths() -> list[str]:
    tree = json.loads(fetch_bytes(JARVIS_TREE_API).decode("utf-8"))["tree"]
    return [item["path"] for item in tree if item.get("type") == "blob"]


def contribution_map(paths: list[str], property_name: str) -> dict[str, str]:
    suffix = f"AI-SinglePropertyPrediction-{property_name}-dft_3d-test-mae.csv.zip"
    return {
        path.split("/contributions/", 1)[1].split("/", 1)[0]: path
        for path in paths
        if path.endswith(suffix) and "/contributions/" in path
    }


def read_zipped_json(path: str) -> dict[str, dict[str, float]]:
    with zipfile.ZipFile(io.BytesIO(fetch_bytes(f"{JARVIS_RAW}/{path}"))) as zf:
        return json.loads(zf.read(zf.namelist()[0]).decode("utf-8"))


def read_zipped_predictions(path: str) -> dict[str, float]:
    with zipfile.ZipFile(io.BytesIO(fetch_bytes(f"{JARVIS_RAW}/{path}"))) as zf:
        lines = zf.read(zf.namelist()[0]).decode("utf-8").strip().splitlines()
    out: dict[str, float] = {}
    for line in lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[0].lower() not in {"id", "jid"}:
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return out


def local_window_certify(
    cal_x: np.ndarray,
    cal_y: np.ndarray,
    test_x: np.ndarray,
    tolerance: float,
    min_support: int = 5,
) -> list[int | None]:
    decisions: list[int | None] = []
    for value in test_x:
        mask = np.abs(cal_x - value) <= tolerance
        if int(mask.sum()) < min_support:
            decisions.append(None)
            continue
        labels = cal_y[mask]
        if bool(labels.all()):
            decisions.append(1)
        elif not bool(labels.any()):
            decisions.append(0)
        else:
            decisions.append(None)
    return decisions


def local_majority(
    cal_x: np.ndarray,
    cal_y: np.ndarray,
    test_x: np.ndarray,
    tolerance: float,
    min_support: int = 5,
) -> list[int]:
    global_majority = int(np.mean(cal_y) >= 0.5)
    decisions: list[int] = []
    for value in test_x:
        mask = np.abs(cal_x - value) <= tolerance
        if int(mask.sum()) < min_support:
            decisions.append(global_majority)
        else:
            decisions.append(int(np.mean(cal_y[mask]) >= 0.5))
    return decisions


def jarvis_replay(n_seeds: int = 50, calibration_fraction: float = 0.5) -> None:
    paths = github_tree_paths()
    formation = contribution_map(paths, "formation_energy_peratom")
    bandgap = contribution_map(paths, "optb88vdw_bandgap")
    shared = sorted(set(formation) & set(bandgap))
    formation_ref = read_zipped_json(
        "jarvis_leaderboard/benchmarks/AI/SinglePropertyPrediction/"
        "dft_3d_formation_energy_peratom.json.zip"
    )["test"]
    bandgap_ref = read_zipped_json(
        "jarvis_leaderboard/benchmarks/AI/SinglePropertyPrediction/"
        "dft_3d_optb88vdw_bandgap.json.zip"
    )["test"]

    rows: list[dict[str, object]] = []
    for model_id in shared:
        formation_pred = read_zipped_predictions(formation[model_id])
        ids = sorted(set(formation_pred) & set(formation_ref) & set(bandgap_ref))
        if len(ids) < 200:
            continue
        pred = np.array([formation_pred[i] for i in ids], dtype=float)
        formation_truth = np.array([formation_ref[i] for i in ids], dtype=float)
        deploy_label = np.array([bandgap_ref[i] > 1.0 for i in ids], dtype=bool)
        n = len(ids)
        benchmark_mae_full = float(np.mean(np.abs(pred - formation_truth)))

        for seed in range(n_seeds):
            rng = np.random.default_rng(9100 + seed)
            order = rng.permutation(n)
            n_cal = int(round(calibration_fraction * n))
            cal, test = order[:n_cal], order[n_cal:]
            tol = float(np.mean(np.abs(pred[cal] - formation_truth[cal])))
            rc = local_window_certify(pred[cal], deploy_label[cal], pred[test], tol)
            maj = local_majority(pred[cal], deploy_label[cal], pred[test], tol)
            global_majority = int(np.mean(deploy_label[cal]) >= 0.5)
            glob = [global_majority for _ in test]

            for policy, decisions in [
                ("response_certified_error_window", rc),
                ("local_majority_certify_all", maj),
                ("global_majority_certify_all", glob),
                (
                    "response_certified_then_acquire_ambiguous",
                    [int(deploy_label[test[j]]) if rc[j] is None else rc[j] for j in range(len(test))],
                ),
            ]:
                certified = [d is not None for d in decisions]
                n_cert = int(sum(certified))
                errors = int(
                    sum(
                        int(decisions[j] != int(deploy_label[test[j]]))
                        for j in range(len(test))
                        if decisions[j] is not None
                    )
                )
                rows.append(
                    {
                        "seed": seed,
                        "model_id": model_id,
                        "policy": policy,
                        "benchmark_mae_full": benchmark_mae_full,
                        "tolerance": tol,
                        "n_test": len(test),
                        "n_certified": n_cert,
                        "certified_fraction": n_cert / len(test),
                        "n_ambiguous": len(test) - n_cert,
                        "error_count": errors,
                        "error_rate_among_certified": errors / n_cert if n_cert else math.nan,
                        "error_rate_all_candidates": errors / len(test),
                        "acquisition_cost": sum(d is None for d in rc)
                        if policy == "response_certified_then_acquire_ambiguous"
                        else 0,
                    }
                )

    write_csv(OUTDIR / "jarvis_operational_replay.csv", rows)
    summarize_policy_rows(rows, OUTDIR / "jarvis_operational_replay_summary.csv")
    summarize_jarvis_model_choice(rows)


def summarize_jarvis_model_choice(rows: list[dict[str, object]]) -> None:
    rc_rows = [r for r in rows if r["policy"] == "response_certified_error_window"]
    out: list[dict[str, object]] = []
    for seed in sorted({int(r["seed"]) for r in rc_rows}):
        trial = [r for r in rc_rows if int(r["seed"]) == seed]
        best_mae = min(trial, key=lambda r: float(r["benchmark_mae_full"]))
        # Prefer lower certified error, then higher certified fraction.
        best_cert = min(
            trial,
            key=lambda r: (
                float(r["error_rate_among_certified"])
                if not math.isnan(float(r["error_rate_among_certified"]))
                else 1.0,
                -float(r["certified_fraction"]),
            ),
        )
        out.append(
            {
                "seed": seed,
                "best_benchmark_model": best_mae["model_id"],
                "best_benchmark_mae": best_mae["benchmark_mae_full"],
                "best_benchmark_certified_fraction": best_mae["certified_fraction"],
                "best_benchmark_certified_error": best_mae["error_rate_among_certified"],
                "best_certification_model": best_cert["model_id"],
                "best_certification_mae": best_cert["benchmark_mae_full"],
                "best_certification_fraction": best_cert["certified_fraction"],
                "best_certification_error": best_cert["error_rate_among_certified"],
                "model_choice_changed": best_mae["model_id"] != best_cert["model_id"],
            }
        )
    write_csv(OUTDIR / "jarvis_operational_model_choice.csv", out)


def plot_operational_replays() -> None:
    tox = pd.read_csv(OUTDIR / "tox21_operational_replay_summary.csv")
    jar = pd.read_csv(OUTDIR / "jarvis_operational_replay_summary.csv")
    mc = pd.read_csv(OUTDIR / "jarvis_operational_model_choice.csv")

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0), constrained_layout=True)

    def bar_panel(ax, df, title, policies):
        sub = df.set_index("policy").loc[policies]
        x = np.arange(len(sub))
        ax.bar(x - 0.18, sub["mean_error_rate_all_candidates"] * 100, width=0.36, color="#c43b3b", label="false decisions")
        ax.bar(x + 0.18, sub["mean_certified_fraction"] * 100, width=0.36, color="#188a5b", label="immediate certifications")
        ax.set_xticks(x)
        ax.set_xticklabels(["benchmark\nmajority", "response-\ncertified", "certify +\nacquire"], fontsize=8)
        ax.set_ylabel("% of held-out candidates")
        ax.set_title(title)
        ax.grid(axis="y", color="#d9dde3", lw=0.7, alpha=0.8)

    bar_panel(
        axes[0],
        tox,
        "Tox21 exact-fiber replay",
        [
            "benchmark_fiber_majority_certify_all",
            "response_certified_exact_fiber",
            "response_certified_then_acquire_ambiguous",
        ],
    )
    bar_panel(
        axes[1],
        jar,
        "JARVIS error-window replay",
        [
            "local_majority_certify_all",
            "response_certified_error_window",
            "response_certified_then_acquire_ambiguous",
        ],
    )

    axes[2].bar(
        ["best\nbenchmark MAE", "best\ncertification"],
        [
            mc["best_benchmark_certified_error"].mean() * 100,
            mc["best_certification_error"].mean() * 100,
        ],
        color=["#6f7f95", "#188a5b"],
    )
    axes[2].set_title("JARVIS model choice replay")
    axes[2].set_ylabel("error among certified candidates (%)")
    axes[2].grid(axis="y", color="#d9dde3", lw=0.7, alpha=0.8)
    axes[2].text(
        0.5,
        0.95,
        f"model choice changed in {mc['model_choice_changed'].mean():.0%} of splits",
        transform=axes[2].transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.37, -0.03))
    fig.savefig(OUTDIR / "operational_decision_replay.png", dpi=300)
    fig.savefig(OUTDIR / "operational_decision_replay.svg")
    plt.close(fig)


def main() -> None:
    tox21_replay()
    jarvis_replay()
    plot_operational_replays()


if __name__ == "__main__":
    main()
