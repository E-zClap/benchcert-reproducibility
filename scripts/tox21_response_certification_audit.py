"""Response-certification audit for MoleculeNet Tox21.

Data source:
* MoleculeNet Tox21 CSV hosted by DeepChem.

Audit:
* benchmark evidence: seven nuclear-receptor assay labels;
* deployment response: SR-p53 stress-response activity label;
* certification unit: exact fibers of identical benchmark-assay labels.

The audit asks whether a standard assay panel certifies a held-out toxicity
endpoint. It is a fixed-label-fiber certification diagnostic, not a new Tox21
leaderboard metric.
"""

from __future__ import annotations

import csv
import gzip
import io
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(exist_ok=True)

TOX21_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
BENCHMARK_ASSAYS = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
]
DEPLOYMENT_ASSAY = "SR-p53"


def fetch_tox21() -> pd.DataFrame:
    request = Request(TOX21_URL, headers={"User-Agent": "Codex"})
    with urlopen(request, timeout=90) as response:
        data = response.read()
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
        return pd.read_csv(gz)


def certification_class(values: pd.Series) -> str:
    labels = set(values.astype(int).tolist())
    if labels == {0}:
        return "certified_viable"
    if labels == {1}:
        return "certified_nonviable"
    return "response_ambiguous"


def main() -> None:
    df = fetch_tox21()
    cols = BENCHMARK_ASSAYS + [DEPLOYMENT_ASSAY, "mol_id", "smiles"]
    audit = df[cols].dropna(subset=BENCHMARK_ASSAYS + [DEPLOYMENT_ASSAY]).copy()
    audit[BENCHMARK_ASSAYS + [DEPLOYMENT_ASSAY]] = audit[
        BENCHMARK_ASSAYS + [DEPLOYMENT_ASSAY]
    ].astype(int)

    rows: list[dict[str, object]] = []
    for key, group in audit.groupby(BENCHMARK_ASSAYS, dropna=False):
        label_values = group[DEPLOYMENT_ASSAY]
        cls = certification_class(label_values)
        benchmark_pattern = "|".join(str(int(v)) for v in key)
        rows.append(
            {
                "benchmark_pattern": benchmark_pattern,
                "n_compounds": len(group),
                "deployment_active": int(label_values.sum()),
                "deployment_inactive": int((label_values == 0).sum()),
                "certification_class": cls,
            }
        )

    fiber_path = OUTDIR / "tox21_response_certification_audit.csv"
    with fiber_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "benchmark_pattern",
            "n_compounds",
            "deployment_active",
            "deployment_inactive",
            "certification_class",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    pattern_to_class = {row["benchmark_pattern"]: row["certification_class"] for row in rows}
    patterns = audit[BENCHMARK_ASSAYS].astype(str).agg("|".join, axis=1)
    candidate_classes = patterns.map(pattern_to_class)
    n_candidates = len(audit)
    summary = {
        "source": "MoleculeNet Tox21 public DeepChem CSV",
        "benchmark_assays": ";".join(BENCHMARK_ASSAYS),
        "deployment_assay": DEPLOYMENT_ASSAY,
        "n_candidates": n_candidates,
        "n_fibers": len(rows),
        "ambiguous_fibers": sum(
            row["certification_class"] == "response_ambiguous" for row in rows
        ),
        "certified_viable_candidates": int((candidate_classes == "certified_viable").sum()),
        "certified_nonviable_candidates": int(
            (candidate_classes == "certified_nonviable").sum()
        ),
        "response_ambiguous_candidates": int(
            (candidate_classes == "response_ambiguous").sum()
        ),
        "ambiguous_candidate_fraction": float(
            (candidate_classes == "response_ambiguous").sum() / n_candidates
        ),
        "deployment_active_prevalence": float(audit[DEPLOYMENT_ASSAY].mean()),
    }

    summary_path = OUTDIR / "tox21_response_certification_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)


if __name__ == "__main__":
    main()
