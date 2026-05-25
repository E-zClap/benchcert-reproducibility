"""Response-certification audit for Matbench Discovery public predictions.

Data sources:
* WBM summary from janosh/matbench-discovery GitHub repository.
* Public model prediction files from the Matbench Discovery Figshare record
  "Model Predictions for Discovery" (article 28187990).

Audit:
* benchmark evidence: model-predicted formation energy per atom;
* deployment claim: thermodynamic stability label from reference
  energy-above-hull (stable if e_above_hull_wbm <= 0 eV/atom);
* certification unit: local fibers of similar predicted formation energy.

This tests whether an accurate formation-energy benchmark prediction certifies
the deployment stability claim. It is a task-level certification audit, not a
new Matbench Discovery metric replacement.
"""

from __future__ import annotations

import csv
import gzip
import io
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(exist_ok=True)

FIGSHARE_API = "https://api.figshare.com/v2/articles/28187990"
WBM_SUMMARY_URL = (
    "https://raw.githubusercontent.com/janosh/matbench-discovery/main/"
    "data/wbm/2023-12-13-wbm-summary.csv.gz"
)


@dataclass(frozen=True)
class AuditRow:
    model_id: str
    n_candidates: int
    formation_energy_mae: float
    stable_prevalence: float
    ambiguous_fibers: int
    total_fibers: int
    ambiguous_fraction: float
    certifiable_candidates: int
    certifiable_fraction: float


def fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Codex"})
    with urlopen(request, timeout=90) as response:
        return response.read()


def model_id_from_name(name: str) -> str:
    clean = name.replace("\\", "/").strip("/")
    parts = clean.split("/")
    if "models" in parts:
        idx = parts.index("models")
        if idx + 2 < len(parts):
            return f"{parts[idx + 1]}:{parts[idx + 2]}"
    stem = parts[-1].replace(".csv.gz", "")
    stem = re.sub(r"^\d{4}-\d{1,2}-\d{1,2}-", "", stem)
    stem = stem.replace("-wbm-IS2RE-FIRE", "").replace("-wbm-IS2RE", "")
    stem = stem.replace("-preds", "")
    return stem


def read_prediction_file(download_url: str) -> pd.DataFrame:
    data = fetch_bytes(download_url)
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
        return pd.read_csv(gz)


def prediction_column(df: pd.DataFrame) -> str:
    candidates = [
        col
        for col in df.columns
        if col != "material_id" and not str(col).lower().startswith("unnamed")
    ]
    numeric = [col for col in candidates if pd.api.types.is_numeric_dtype(df[col])]
    if not numeric:
        raise ValueError("no numeric prediction column found")
    eform_cols = [col for col in numeric if "e_form" in str(col)]
    return eform_cols[0] if eform_cols else numeric[0]


def fiber_audit(
    pred: np.ndarray,
    stable: np.ndarray,
    n_bins: int = 20,
) -> tuple[int, int, int]:
    quantiles = np.unique(np.quantile(pred, np.linspace(0.0, 1.0, n_bins + 1)))
    ambiguous = 0
    total = 0
    certifiable_candidates = 0
    for lo, hi in zip(quantiles[:-1], quantiles[1:]):
        if hi == quantiles[-1]:
            mask = (pred >= lo) & (pred <= hi)
        else:
            mask = (pred >= lo) & (pred < hi)
        if int(mask.sum()) < 10:
            continue
        labels = stable[mask]
        total += 1
        if bool(labels.any()) and not bool(labels.all()):
            ambiguous += 1
        else:
            certifiable_candidates += int(mask.sum())
    return ambiguous, total, certifiable_candidates


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    wbm = pd.read_csv(io.BytesIO(fetch_bytes(WBM_SUMMARY_URL)), compression="gzip")
    wbm = wbm[["material_id", "e_form_per_atom_wbm", "e_above_hull_wbm"]].dropna()
    wbm["stable"] = wbm["e_above_hull_wbm"] <= 0.0

    import json

    file_items = json.loads(fetch_bytes(FIGSHARE_API).decode("utf-8"))["files"]
    seen_models: set[str] = set()
    rows: list[AuditRow] = []
    errors: list[dict[str, str]] = []

    for item in file_items:
        name = item["name"]
        if not name.endswith(".csv.gz"):
            continue
        model_id = model_id_from_name(name)
        if model_id in seen_models:
            continue
        seen_models.add(model_id)
        try:
            pred_df = read_prediction_file(item["download_url"])
            pred_col = prediction_column(pred_df)
            merged = wbm.merge(
                pred_df[["material_id", pred_col]].dropna(),
                on="material_id",
                how="inner",
            )
            if len(merged) < 100:
                continue
            pred = merged[pred_col].to_numpy(dtype=float)
            truth = merged["e_form_per_atom_wbm"].to_numpy(dtype=float)
            stable = merged["stable"].to_numpy(dtype=bool)
            ambiguous, total_fibers, certifiable = fiber_audit(pred, stable)
            if total_fibers == 0:
                continue
            rows.append(
                AuditRow(
                    model_id=model_id,
                    n_candidates=int(len(merged)),
                    formation_energy_mae=float(np.mean(np.abs(pred - truth))),
                    stable_prevalence=float(stable.mean()),
                    ambiguous_fibers=ambiguous,
                    total_fibers=total_fibers,
                    ambiguous_fraction=ambiguous / total_fibers,
                    certifiable_candidates=certifiable,
                    certifiable_fraction=certifiable / len(merged),
                )
            )
        except Exception as exc:  # pragma: no cover - recorded for auditability
            errors.append({"file": name, "error": str(exc)})

    out_rows = [row.__dict__ for row in sorted(rows, key=lambda r: r.formation_energy_mae)]
    write_csv(OUTDIR / "matbench_discovery_certification_audit.csv", out_rows)
    if errors:
        write_csv(OUTDIR / "matbench_discovery_certification_errors.csv", errors)

    summary = [
        {
            "source": "Matbench Discovery public Figshare predictions",
            "models_audited": len(out_rows),
            "median_formation_energy_mae": float(
                np.median([r["formation_energy_mae"] for r in out_rows])
            )
            if out_rows
            else np.nan,
            "median_ambiguous_fraction": float(
                np.median([r["ambiguous_fraction"] for r in out_rows])
            )
            if out_rows
            else np.nan,
            "median_certifiable_fraction": float(
                np.median([r["certifiable_fraction"] for r in out_rows])
            )
            if out_rows
            else np.nan,
        }
    ]
    write_csv(OUTDIR / "matbench_discovery_certification_summary.csv", summary)

    if out_rows:
        fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
        ax.scatter(
            [r["formation_energy_mae"] for r in out_rows],
            [r["ambiguous_fraction"] for r in out_rows],
            s=54,
            color="#335c9d",
            alpha=0.78,
        )
        for r in out_rows[:6]:
            ax.annotate(
                r["model_id"],
                (r["formation_energy_mae"], r["ambiguous_fraction"]),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )
        ax.set_xlabel("formation-energy MAE (eV/atom)")
        ax.set_ylabel("ambiguous local stability-fiber fraction")
        ax.set_title("Matbench Discovery: accuracy vs certification")
        fig.savefig(OUTDIR / "matbench_discovery_accuracy_vs_certification.svg")
        fig.savefig(OUTDIR / "matbench_discovery_accuracy_vs_certification.png", dpi=200)


if __name__ == "__main__":
    main()
