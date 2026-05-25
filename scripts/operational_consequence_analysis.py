"""Costed operational consequence analysis for response-rank replays.

The operational replay files report false deployment decisions and the number
of ambiguous cases that would be sent to acquisition.  This script converts
those counts into a simple decision-cost analysis:

* benchmark-only: act on the benchmark-response majority for every candidate;
* certify+acquire: certify only determined fibers and measure the deployment
  response for ambiguous fibers.

Costs are expressed in units of one false deployment decision.  The break-even
acquisition cost is the largest per-candidate measurement cost for which
certify+acquire is cheaper than acting on benchmark evidence alone.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def paired_cost_rows(
    df: pd.DataFrame,
    *,
    domain: str,
    id_cols: list[str],
    baseline_policy: str,
    acquire_policy: str,
    acquisition_cost_grid: list[float],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    base = df[df["policy"] == baseline_policy].copy()
    acq = df[df["policy"] == acquire_policy].copy()
    merged = base.merge(acq, on=id_cols, suffixes=("_baseline", "_acquire"))

    pair_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        baseline_errors = float(row["error_count_baseline"])
        acquire_errors = float(row["error_count_acquire"])
        acquisitions = float(row["acquisition_cost_acquire"])
        avoided = baseline_errors - acquire_errors
        break_even = avoided / acquisitions if acquisitions > 0 else np.nan
        pair = {
            "domain": domain,
            **{col: row[col] for col in id_cols},
            "baseline_policy": baseline_policy,
            "acquire_policy": acquire_policy,
            "n_test": int(row["n_test_baseline"]),
            "baseline_false_decisions": baseline_errors,
            "acquire_false_decisions": acquire_errors,
            "false_decisions_avoided": avoided,
            "acquisitions": acquisitions,
            "break_even_acquisition_cost": break_even,
            "baseline_error_rate": float(row["error_rate_all_candidates_baseline"]),
            "acquire_error_rate": float(row["error_rate_all_candidates_acquire"]),
        }
        pair_rows.append(pair)
        for cost in acquisition_cost_grid:
            baseline_total = baseline_errors
            acquire_total = acquire_errors + cost * acquisitions
            curve_rows.append(
                {
                    "domain": domain,
                    **{col: row[col] for col in id_cols},
                    "acquisition_cost_per_candidate": cost,
                    "baseline_total_cost": baseline_total,
                    "certify_acquire_total_cost": acquire_total,
                    "relative_cost_reduction": 1.0 - acquire_total / baseline_total
                    if baseline_total > 0
                    else np.nan,
                }
            )
    return pair_rows, curve_rows


def summarize_break_even(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for domain in sorted({str(r["domain"]) for r in rows}):
        sub = [r for r in rows if str(r["domain"]) == domain]
        be = np.array([float(r["break_even_acquisition_cost"]) for r in sub], dtype=float)
        avoided = np.array([float(r["false_decisions_avoided"]) for r in sub], dtype=float)
        baseline = np.array([float(r["baseline_false_decisions"]) for r in sub], dtype=float)
        acquire = np.array([float(r["acquire_false_decisions"]) for r in sub], dtype=float)
        acq = np.array([float(r["acquisitions"]) for r in sub], dtype=float)
        out.append(
            {
                "domain": domain,
                "n_replays": len(sub),
                "mean_baseline_false_decisions": float(np.mean(baseline)),
                "mean_certify_acquire_false_decisions": float(np.mean(acquire)),
                "mean_false_decisions_avoided": float(np.mean(avoided)),
                "mean_acquisitions": float(np.mean(acq)),
                "mean_break_even_acquisition_cost": float(np.nanmean(be)),
                "median_break_even_acquisition_cost": float(np.nanmedian(be)),
                "p10_break_even_acquisition_cost": float(np.nanpercentile(be, 10)),
                "p90_break_even_acquisition_cost": float(np.nanpercentile(be, 90)),
            }
        )
    return out


def summarize_cost_curve(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    df = pd.DataFrame(rows)
    for (domain, cost), sub in df.groupby(["domain", "acquisition_cost_per_candidate"]):
        rel = sub["relative_cost_reduction"].to_numpy(dtype=float)
        out.append(
            {
                "domain": domain,
                "acquisition_cost_per_candidate": float(cost),
                "mean_relative_cost_reduction": float(np.nanmean(rel)),
                "p10_relative_cost_reduction": float(np.nanpercentile(rel, 10)),
                "p90_relative_cost_reduction": float(np.nanpercentile(rel, 90)),
            }
        )
    return out


def jarvis_case_study() -> list[dict[str, object]]:
    replay = pd.read_csv(OUT / "jarvis_operational_replay.csv")
    choice = pd.read_csv(OUT / "jarvis_operational_model_choice.csv")
    # Choose the split where certification most improves certified error over
    # the benchmark-MAE-selected model.
    choice = choice.copy()
    choice["error_gap"] = (
        choice["best_benchmark_certified_error"].astype(float)
        - choice["best_certification_error"].astype(float)
    )
    selected = choice.sort_values(["error_gap", "best_certification_fraction"], ascending=False).iloc[0]
    seed = int(selected["seed"])
    models = [selected["best_benchmark_model"], selected["best_certification_model"]]
    rows: list[dict[str, object]] = []
    for model in models:
        sub = replay[(replay["seed"] == seed) & (replay["model_id"] == model)]
        for policy in [
            "local_majority_certify_all",
            "response_certified_error_window",
            "response_certified_then_acquire_ambiguous",
        ]:
            r = sub[sub["policy"] == policy].iloc[0]
            rows.append(
                {
                    "seed": seed,
                    "role": "benchmark_mae_choice"
                    if model == selected["best_benchmark_model"]
                    else "certification_choice",
                    "model_id": model,
                    "policy": policy,
                    "benchmark_mae": float(r["benchmark_mae_full"]),
                    "n_test": int(r["n_test"]),
                    "certified_fraction": float(r["certified_fraction"]),
                    "false_decisions": int(r["error_count"]),
                    "false_decision_rate_all": float(r["error_rate_all_candidates"]),
                    "error_rate_among_certified": float(r["error_rate_among_certified"]),
                    "acquisitions": int(r["acquisition_cost"]),
                }
            )
    return rows


def main() -> None:
    grid = [0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20, 0.50]
    all_pairs: list[dict[str, object]] = []
    all_curves: list[dict[str, object]] = []

    tox = pd.read_csv(OUT / "tox21_operational_replay.csv")
    pairs, curves = paired_cost_rows(
        tox,
        domain="Tox21",
        id_cols=["seed"],
        baseline_policy="benchmark_fiber_majority_certify_all",
        acquire_policy="response_certified_then_acquire_ambiguous",
        acquisition_cost_grid=grid,
    )
    all_pairs.extend(pairs)
    all_curves.extend(curves)

    jarvis = pd.read_csv(OUT / "jarvis_operational_replay.csv")
    pairs, curves = paired_cost_rows(
        jarvis,
        domain="JARVIS",
        id_cols=["seed", "model_id"],
        baseline_policy="local_majority_certify_all",
        acquire_policy="response_certified_then_acquire_ambiguous",
        acquisition_cost_grid=grid,
    )
    all_pairs.extend(pairs)
    all_curves.extend(curves)

    write_csv(OUT / "operational_cost_break_even.csv", all_pairs)
    break_summary = summarize_break_even(all_pairs)
    write_csv(OUT / "operational_cost_break_even_summary.csv", break_summary)
    write_csv(OUT / "operational_cost_curve.csv", all_curves)
    curve_summary = summarize_cost_curve(all_curves)
    write_csv(OUT / "operational_cost_curve_summary.csv", curve_summary)
    case = jarvis_case_study()
    write_csv(OUT / "jarvis_operational_case_study.csv", case)

    print("Break-even summary:")
    for row in break_summary:
        print(row)
    print("Cost curve summary:")
    for row in curve_summary:
        print(row)
    print("JARVIS case study:")
    for row in case:
        print(row)


if __name__ == "__main__":
    main()
