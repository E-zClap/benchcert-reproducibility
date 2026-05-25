"""Real public-prediction audit for response certification on JARVIS-Leaderboard.

The audit uses public benchmark targets and submitted model prediction files
from the usnistgov/jarvis_leaderboard GitHub repository. It deliberately avoids
claiming a physical deployment benchmark that the source data do not define.
Instead, it runs a cross-property certification stress test:

* benchmark evidence: a model's formation-energy prediction;
* deployment response: optB88vdW band gap threshold on the same materials;
* certification unit: a local fiber of candidates with similar benchmark
  predictions.

If a local fiber contains both band-gap labels, a benchmark-only claim is
response-ambiguous for that deployment threshold. This is an empirical
fixed-label-fiber analogue of the manuscript's certification trichotomy.
"""

from __future__ import annotations

import csv
import io
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(exist_ok=True)

RAW_BASE = "https://raw.githubusercontent.com/usnistgov/jarvis_leaderboard/main"
TREE_API = "https://api.github.com/repos/usnistgov/jarvis_leaderboard/git/trees/main?recursive=1"


@dataclass(frozen=True)
class ModelAudit:
    model_id: str
    benchmark_mae: float
    deployment_mae: float
    n_overlap: int
    ambiguous_fibers: int
    total_fibers: int
    ambiguous_fraction: float
    certifiable_candidates: int
    total_candidates: int
    certifiable_fraction: float


def fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Codex"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def read_zipped_json(path: str) -> dict[str, dict[str, float]]:
    data = fetch_bytes(f"{RAW_BASE}/{path}")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        name = zf.namelist()[0]
        return json.loads(zf.read(name).decode("utf-8"))


def read_zipped_predictions(path: str) -> dict[str, float]:
    data = fetch_bytes(f"{RAW_BASE}/{path}")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        name = zf.namelist()[0]
        text = zf.read(name).decode("utf-8").strip().splitlines()
    rows: dict[str, float] = {}
    for line in text:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or parts[0].lower() in {"id", "jid"}:
            continue
        try:
            rows[parts[0]] = float(parts[1])
        except ValueError:
            continue
    return rows


def github_tree_paths() -> list[str]:
    tree = json.loads(fetch_bytes(TREE_API).decode("utf-8"))["tree"]
    return [item["path"] for item in tree if item.get("type") == "blob"]


def contribution_map(paths: list[str], property_name: str) -> dict[str, str]:
    suffix = f"AI-SinglePropertyPrediction-{property_name}-dft_3d-test-mae.csv.zip"
    found: dict[str, str] = {}
    for path in paths:
        if path.endswith(suffix) and "/contributions/" in path:
            model_id = path.split("/contributions/", 1)[1].split("/", 1)[0]
            found[model_id] = path
    return found


def mae(pred: dict[str, float], truth: dict[str, float], ids: list[str]) -> float:
    return float(np.mean([abs(pred[i] - truth[i]) for i in ids]))


def fiber_audit(
    benchmark_values: dict[str, float],
    deployment_truth: dict[str, float],
    ids: list[str],
    threshold: float,
    n_bins: int = 20,
) -> tuple[int, int, int, int]:
    values = np.array([benchmark_values[i] for i in ids], dtype=float)
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    quantiles = np.unique(quantiles)
    if len(quantiles) <= 2:
        return 0, 0, 0, len(ids)

    ambiguous_fibers = 0
    total_fibers = 0
    certifiable_candidates = 0
    for lo, hi in zip(quantiles[:-1], quantiles[1:]):
        if hi == quantiles[-1]:
            selected = [i for i in ids if lo <= benchmark_values[i] <= hi]
        else:
            selected = [i for i in ids if lo <= benchmark_values[i] < hi]
        if len(selected) < 5:
            continue
        labels = [deployment_truth[i] > threshold for i in selected]
        total_fibers += 1
        if any(labels) and not all(labels):
            ambiguous_fibers += 1
        else:
            certifiable_candidates += len(selected)
    return ambiguous_fibers, total_fibers, certifiable_candidates, len(ids)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    benchmark_property = "formation_energy_peratom"
    deployment_property = "optb88vdw_bandgap"
    deployment_threshold = 1.0

    paths = github_tree_paths()
    formation_contribs = contribution_map(paths, benchmark_property)
    bandgap_contribs = contribution_map(paths, deployment_property)
    shared_models = sorted(set(formation_contribs) & set(bandgap_contribs))

    formation_ref = read_zipped_json(
        "jarvis_leaderboard/benchmarks/AI/SinglePropertyPrediction/"
        "dft_3d_formation_energy_peratom.json.zip"
    )["test"]
    bandgap_ref = read_zipped_json(
        "jarvis_leaderboard/benchmarks/AI/SinglePropertyPrediction/"
        "dft_3d_optb88vdw_bandgap.json.zip"
    )["test"]

    audits: list[ModelAudit] = []
    fiber_rows: list[dict[str, object]] = []
    for model_id in shared_models:
        formation_pred = read_zipped_predictions(formation_contribs[model_id])
        bandgap_pred = read_zipped_predictions(bandgap_contribs[model_id])
        ids = sorted(
            set(formation_pred)
            & set(bandgap_pred)
            & set(formation_ref)
            & set(bandgap_ref)
        )
        if len(ids) < 50:
            continue
        ambiguous, total_fibers, certifiable, total = fiber_audit(
            formation_pred,
            bandgap_ref,
            ids,
            deployment_threshold,
        )
        if total_fibers == 0:
            continue
        audits.append(
            ModelAudit(
                model_id=model_id,
                benchmark_mae=mae(formation_pred, formation_ref, ids),
                deployment_mae=mae(bandgap_pred, bandgap_ref, ids),
                n_overlap=len(ids),
                ambiguous_fibers=ambiguous,
                total_fibers=total_fibers,
                ambiguous_fraction=ambiguous / total_fibers,
                certifiable_candidates=certifiable,
                total_candidates=total,
                certifiable_fraction=certifiable / total,
            )
        )
        fiber_rows.append(
            {
                "model_id": model_id,
                "benchmark_property": benchmark_property,
                "deployment_property": deployment_property,
                "deployment_threshold": deployment_threshold,
                "n_overlap": len(ids),
                "benchmark_mae": audits[-1].benchmark_mae,
                "deployment_mae": audits[-1].deployment_mae,
                "ambiguous_fibers": ambiguous,
                "total_fibers": total_fibers,
                "ambiguous_fraction": audits[-1].ambiguous_fraction,
                "certifiable_candidates": certifiable,
                "total_candidates": total,
                "certifiable_fraction": audits[-1].certifiable_fraction,
            }
        )

    rows = [audit.__dict__ for audit in sorted(audits, key=lambda x: x.benchmark_mae)]
    write_csv(OUTDIR / "jarvis_public_model_certification_audit.csv", rows)

    summary = {
        "source": "JARVIS-Leaderboard public GitHub predictions",
        "benchmark_property": benchmark_property,
        "deployment_property": deployment_property,
        "deployment_threshold": deployment_threshold,
        "models_with_both_prediction_files": len(shared_models),
        "models_audited": len(rows),
        "median_ambiguous_fraction": float(np.median([r["ambiguous_fraction"] for r in rows]))
        if rows
        else math.nan,
        "median_certifiable_fraction": float(np.median([r["certifiable_fraction"] for r in rows]))
        if rows
        else math.nan,
    }
    write_csv(OUTDIR / "jarvis_public_model_certification_summary.csv", [summary])

    if rows:
        fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
        x = [r["benchmark_mae"] for r in rows]
        y = [r["ambiguous_fraction"] for r in rows]
        sizes = [max(24, min(140, r["n_overlap"] / 20)) for r in rows]
        ax.scatter(x, y, s=sizes, color="#335c9d", alpha=0.78)
        for r in rows[:5]:
            ax.annotate(
                r["model_id"],
                (r["benchmark_mae"], r["ambiguous_fraction"]),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )
        ax.set_xlabel("formation-energy test MAE")
        ax.set_ylabel("ambiguous local deployment-fiber fraction")
        ax.set_title("JARVIS public predictions: accuracy vs certification")
        fig.savefig(OUTDIR / "jarvis_accuracy_vs_certification.svg")
        fig.savefig(OUTDIR / "jarvis_accuracy_vs_certification.png", dpi=200)


if __name__ == "__main__":
    main()
