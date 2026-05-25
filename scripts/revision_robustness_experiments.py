"""Revision robustness experiments for response-rank certification.

This script adds the checks requested during review:

* repeated-seed conformal transfer statistics;
* repeated structured and null leaderboard simulations;
* bin-count and nearest-neighbor sensitivity for Matbench Discovery fibers;
* bin-count and nearest-neighbor sensitivity for JARVIS cross-property fibers.

The outputs are compact CSV summaries used by the revised manuscript.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(exist_ok=True)
CACHEDIR = ROOT / ".cache" / "downloads"
CACHEDIR.mkdir(parents=True, exist_ok=True)

FIGSHARE_API = "https://api.figshare.com/v2/articles/28187990"
WBM_SUMMARY_URL = (
    "https://raw.githubusercontent.com/janosh/matbench-discovery/main/"
    "data/wbm/2023-12-13-wbm-summary.csv.gz"
)
JARVIS_RAW = "https://raw.githubusercontent.com/usnistgov/jarvis_leaderboard/main"
JARVIS_TREE_API = (
    "https://api.github.com/repos/usnistgov/jarvis_leaderboard/git/trees/main?recursive=1"
)


def fetch_bytes(url: str, timeout: int = 90, use_cache: bool = True) -> bytes:
    cache_path = CACHEDIR / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()}.bin"
    if use_cache and cache_path.exists():
        return cache_path.read_bytes()
    request = Request(url, headers={"User-Agent": "Codex"})
    with urlopen(request, timeout=timeout) as response:
        data = response.read()
    if use_cache:
        cache_path.write_bytes(data)
    return data


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def row_span_projector(rows: np.ndarray) -> np.ndarray:
    if rows.ndim == 1:
        rows = rows[np.newaxis]
    q, _ = np.linalg.qr(rows.T)
    rank = np.linalg.matrix_rank(rows, tol=1e-9)
    return q[:, :rank] @ q[:, :rank].T


def residual_norm(rows: np.ndarray, probe: np.ndarray) -> float:
    p = row_span_projector(rows)
    return float(np.linalg.norm(probe - p @ probe))


def make_geometry(seed: int) -> tuple[np.ndarray, np.ndarray]:
    d = 8
    rng = np.random.default_rng(seed)
    e = np.eye(d)

    b = np.array(
        [
            e[0] + 0.05 * rng.standard_normal(d),
            e[1] + 0.05 * rng.standard_normal(d),
            0.6 * e[0] + 0.8 * e[1] + 0.04 * rng.standard_normal(d),
            0.4 * e[1] + 0.9 * e[2] + 0.04 * rng.standard_normal(d),
        ]
    )
    b = b / np.linalg.norm(b, axis=1, keepdims=True)

    deploy = np.array(
        [
            0.90 * e[5] + 0.30 * e[0] + 0.15 * e[1],
            0.88 * e[6] + 0.28 * e[1] + 0.14 * e[2],
            0.87 * e[7] + 0.40 * e[0] + 0.10 * e[2],
        ]
    )
    deploy = deploy / np.linalg.norm(deploy, axis=1, keepdims=True)
    return b, deploy


def conformal_once(seed: int) -> dict[str, float]:
    b, deploy = make_geometry(1000 + seed)
    rng = np.random.default_rng(2000 + seed)
    d = 8
    sigma = 0.05
    r_cert = 2.0
    n = 600
    n_train, n_cal = 200, 200

    x = rng.standard_normal((n, d))
    p_b = row_span_projector(b)
    x_vis = x @ p_b
    k_deploy = deploy[seed % len(deploy)]
    g = residual_norm(b, k_deploy)

    y_bench = x @ b[0] + sigma * rng.standard_normal(n)
    y_deploy = x @ k_deploy + sigma * rng.standard_normal(n)
    idx = rng.permutation(n)
    tr = idx[:n_train]
    cal = idx[n_train : n_train + n_cal]
    te = idx[n_train + n_cal :]

    w_bench, *_ = np.linalg.lstsq(x_vis[tr], y_bench[tr], rcond=None)
    w_deploy, *_ = np.linalg.lstsq(x_vis[tr], y_deploy[tr], rcond=None)
    yp_bench_cal = x_vis[cal] @ w_bench
    yp_bench_te = x_vis[te] @ w_bench
    yp_deploy_cal = x_vis[cal] @ w_deploy
    yp_deploy_te = x_vis[te] @ w_deploy

    alpha = 0.05
    q_bench = np.quantile(
        np.abs(yp_bench_cal - y_bench[cal]),
        min(1.0, (1.0 - alpha) * (len(cal) + 1) / len(cal)),
    )
    q_deploy = np.quantile(
        np.abs(yp_deploy_cal - y_deploy[cal]),
        min(1.0, (1.0 - alpha) * (len(cal) + 1) / len(cal)),
    )

    bench_cov = float(np.mean(np.abs(yp_bench_te - y_bench[te]) <= q_bench))
    transfer_cov = float(np.mean(np.abs(yp_deploy_te - y_deploy[te]) <= q_bench))
    oracle_cov = float(np.mean(np.abs(yp_deploy_te - y_deploy[te]) <= q_deploy))
    cert_cov = float(np.mean(np.abs(yp_deploy_te - y_deploy[te]) <= r_cert * g))

    return {
        "seed": seed,
        "deployment_residual": g,
        "benchmark_conformal_coverage": bench_cov,
        "benchmark_transfer_coverage": transfer_cov,
        "oracle_deployment_coverage": oracle_cov,
        "response_rank_coverage": cert_cov,
        "benchmark_width": 2.0 * float(q_bench),
        "oracle_width": 2.0 * float(q_deploy),
        "response_rank_width": 2.0 * r_cert * g,
    }


def summarize_numeric(rows: list[dict[str, float]], keys: list[str]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    n = len(rows)
    for key in keys:
        vals = np.array([float(row[key]) for row in rows], dtype=float)
        se = float(vals.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
        out.append(
            {
                "metric": key,
                "n_runs": n,
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if n > 1 else 0.0,
                "ci95_low": float(vals.mean() - 1.96 * se) if n > 1 else math.nan,
                "ci95_high": float(vals.mean() + 1.96 * se) if n > 1 else math.nan,
            }
        )
    return out


def model_certification_trial(seed: int, structured: bool) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    d = 10
    n_candidates = 500
    top_k = 100
    e = np.eye(d)
    k_star = sum(e[i] for i in range(8)) / np.sqrt(8.0)
    x = rng.standard_normal((n_candidates, d))
    r_c = rng.uniform(1.2, 2.8, size=n_candidates)
    tau = 0.0
    ranks = [3, 4, 6, 7, 8]
    structured_noise = [0.04, 0.07, 0.11, 0.16, 0.22]
    null_noise = [0.10, 0.10, 0.10, 0.10, 0.10]
    noises = structured_noise if structured else null_noise

    rows: list[dict[str, object]] = []
    for model_idx, (rank, noise) in enumerate(zip(ranks, noises)):
        probes = []
        for j in range(rank):
            probes.append(e[j] + (0.03 + 0.01 * model_idx) * rng.standard_normal(d))
        b = np.array(probes)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
        p = row_span_projector(b)
        g = float(np.linalg.norm(k_star - p @ k_star))

        scores = x @ p @ k_star + noise * rng.standard_normal(n_candidates)
        bench_dir = b[0] / (np.linalg.norm(b[0]) + 1e-12)
        y_bench = x @ bench_dir
        y_pred = x @ p @ bench_dir + noise * rng.standard_normal(n_candidates)
        mae = float(np.mean(np.abs(y_pred - y_bench)))
        top = np.argsort(scores)[:top_k]
        certified = int(np.sum(scores[top] + r_c[top] * g < tau))
        rows.append(
            {
                "seed": seed,
                "setting": "structured" if structured else "null_equal_noise",
                "model": chr(ord("A") + model_idx),
                "span_rank": rank,
                "benchmark_mae": mae,
                "deployment_residual": g,
                "certified_fraction": certified / top_k,
            }
        )
    return rows


def leaderboard_repeats(n_runs: int = 50) -> None:
    rows: list[dict[str, object]] = []
    for seed in range(n_runs):
        rows.extend(model_certification_trial(3000 + seed, structured=True))
        rows.extend(model_certification_trial(4000 + seed, structured=False))
    write_csv(OUTDIR / "revision_leaderboard_repeated_runs.csv", rows)

    summary: list[dict[str, object]] = []
    for setting in ["structured", "null_equal_noise"]:
        sub = [r for r in rows if r["setting"] == setting]
        for model in ["A", "B", "C", "D", "E"]:
            vals = np.array(
                [float(r["certified_fraction"]) for r in sub if r["model"] == model],
                dtype=float,
            )
            maes = np.array(
                [float(r["benchmark_mae"]) for r in sub if r["model"] == model],
                dtype=float,
            )
            summary.append(
                {
                    "setting": setting,
                    "model": model,
                    "n_runs": n_runs,
                    "mean_benchmark_mae": float(maes.mean()),
                    "mean_certified_fraction": float(vals.mean()),
                    "std_certified_fraction": float(vals.std(ddof=1)),
                }
            )

        top_mae_not_top_cert = 0
        for seed in sorted({int(r["seed"]) for r in sub}):
            trial = [r for r in sub if int(r["seed"]) == seed]
            best_mae = min(trial, key=lambda r: float(r["benchmark_mae"]))["model"]
            best_cert = max(trial, key=lambda r: float(r["certified_fraction"]))["model"]
            top_mae_not_top_cert += int(best_mae != best_cert)
        summary.append(
            {
                "setting": setting,
                "model": "best_mae_not_best_cert",
                "n_runs": n_runs,
                "mean_benchmark_mae": math.nan,
                "mean_certified_fraction": top_mae_not_top_cert / n_runs,
                "std_certified_fraction": math.nan,
            }
        )
    write_csv(OUTDIR / "revision_leaderboard_repeated_summary.csv", summary)


def quantile_fiber_stats(values: np.ndarray, labels: np.ndarray, n_bins: int, min_size: int) -> tuple[int, int, int, float]:
    cuts = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
    ambiguous = 0
    total = 0
    certifiable = 0
    sizes: list[int] = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        mask = (values >= lo) & (values <= hi) if hi == cuts[-1] else (values >= lo) & (values < hi)
        if int(mask.sum()) < min_size:
            continue
        lab = labels[mask]
        total += 1
        sizes.append(int(mask.sum()))
        if bool(lab.any()) and not bool(lab.all()):
            ambiguous += 1
        else:
            certifiable += int(mask.sum())
    return ambiguous, total, certifiable, float(np.median(sizes)) if sizes else math.nan


def knn_certifiable_fraction(values: np.ndarray, labels: np.ndarray, k: int) -> tuple[float, float]:
    order = np.argsort(values)
    sorted_labels = labels[order]
    n = len(labels)
    cert = 0
    half = k // 2
    for pos in range(n):
        lo = max(0, pos - half)
        hi = min(n, lo + k)
        lo = max(0, hi - k)
        lab = sorted_labels[lo:hi]
        if bool(lab.all()) or not bool(lab.any()):
            cert += 1
    return cert / n, float(k)


def interval_fiber_stats(
    values: np.ndarray,
    labels: np.ndarray,
    tolerance: float,
) -> tuple[float, float]:
    """Certifiable fraction for |value_i - value_j| <= tolerance.

    O(n log n) due to sorting/searchsorted, instead of O(n^2).
    """
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=bool)

    finite = np.isfinite(values)
    values = values[finite]
    labels = labels[finite]

    n = len(values)
    if n == 0:
        return math.nan, math.nan

    order = np.argsort(values)
    v = values[order]
    y = labels[order].astype(np.int64)

    left = np.searchsorted(v, v - tolerance, side="left")
    right = np.searchsorted(v, v + tolerance, side="right")

    prefix = np.concatenate([[0], np.cumsum(y)])
    counts = right - left
    positives = prefix[right] - prefix[left]

    certifiable = (positives == 0) | (positives == counts)

    return float(np.mean(certifiable)), float(np.median(counts))


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
    return stem.replace("-preds", "")


def prediction_column(df: pd.DataFrame) -> str:
    candidates = [
        col for col in df.columns if col != "material_id" and not str(col).lower().startswith("unnamed")
    ]
    numeric = [col for col in candidates if pd.api.types.is_numeric_dtype(df[col])]
    eform = [col for col in numeric if "e_form" in str(col)]
    return eform[0] if eform else numeric[0]


def matbench_sensitivity(
    max_models: int | None = None,
    model_filter: str | None = None,
    progress: bool = True,
) -> None:
    """Run Matbench Discovery finite-resolution fiber sensitivity.

    Parameters
    ----------
    max_models:
        Optional cap for smoke tests, e.g. ``max_models=2``.
    model_filter:
        Optional substring filter applied to the parsed model id.
    progress:
        Print per-model progress. Useful because public Figshare downloads are
        the slow part of this audit.
    """
    wbm = pd.read_csv(io.BytesIO(fetch_bytes(WBM_SUMMARY_URL)), compression="gzip")
    wbm = wbm[["material_id", "e_above_hull_wbm"]].dropna()
    wbm["stable"] = wbm["e_above_hull_wbm"] <= 0.0
    files = json.loads(fetch_bytes(FIGSHARE_API).decode("utf-8"))["files"]

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    audited = 0
    for item in files:
        name = item["name"]
        if not name.endswith(".csv.gz"):
            continue
        model_id = model_id_from_name(name)
        if model_filter and model_filter not in model_id:
            continue
        if model_id in seen:
            continue
        if max_models is not None and audited >= max_models:
            break
        seen.add(model_id)
        if progress:
            print(f"[matbench] downloading/auditing {audited + 1}: {model_id}", flush=True)
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(fetch_bytes(item["download_url"]))) as gz:
                pred_df = pd.read_csv(gz)
            pred_col = prediction_column(pred_df)
            merged = wbm.merge(pred_df[["material_id", pred_col]].dropna(), on="material_id")
            if len(merged) < 100:
                continue
            pred = merged[pred_col].to_numpy(dtype=float)
            truth = merged["e_above_hull_wbm"].to_numpy(dtype=float)
            stable = merged["stable"].to_numpy(dtype=bool)
            abs_err = np.abs(pred - truth)
            for n_bins in [5, 10, 20, 40]:
                ambiguous, total, cert, median_size = quantile_fiber_stats(pred, stable, n_bins, min_size=10)
                rows.append(
                    {
                        "source": "matbench_discovery",
                        "method": f"quantile_{n_bins}",
                        "model_id": model_id,
                        "n_candidates": len(merged),
                        "ambiguous_fibers": ambiguous,
                        "total_fibers": total,
                        "ambiguous_fraction": ambiguous / total if total else math.nan,
                        "certifiable_fraction": cert / len(merged),
                        "median_fiber_size": median_size,
                        "tolerance": math.nan,
                    }
                )
            for k in [25, 50, 100]:
                cert_frac, median_size = knn_certifiable_fraction(pred, stable, k)
                rows.append(
                    {
                        "source": "matbench_discovery",
                        "method": f"knn_{k}",
                        "model_id": model_id,
                        "n_candidates": len(merged),
                        "ambiguous_fibers": math.nan,
                        "total_fibers": math.nan,
                        "ambiguous_fraction": math.nan,
                        "certifiable_fraction": cert_frac,
                        "median_fiber_size": median_size,
                        "tolerance": math.nan,
                    }
                )
            for method, tolerance in [
                ("interval_mae", float(abs_err.mean())),
                ("interval_q80_abs_error", float(np.quantile(abs_err, 0.80))),
            ]:
                cert_frac, median_size = interval_fiber_stats(pred, stable, tolerance)
                rows.append(
                    {
                        "source": "matbench_discovery",
                        "method": method,
                        "model_id": model_id,
                        "n_candidates": len(merged),
                        "ambiguous_fibers": math.nan,
                        "total_fibers": math.nan,
                        "ambiguous_fraction": math.nan,
                        "certifiable_fraction": cert_frac,
                        "median_fiber_size": median_size,
                        "tolerance": tolerance,
                    }
                )
            audited += 1
        except Exception as exc:
            rows.append(
                {
                    "source": "matbench_discovery",
                    "method": "error",
                    "model_id": model_id,
                    "n_candidates": 0,
                    "ambiguous_fibers": 0,
                    "total_fibers": 0,
                    "ambiguous_fraction": math.nan,
                    "certifiable_fraction": math.nan,
                    "median_fiber_size": math.nan,
                    "tolerance": math.nan,
                    "error": str(exc),
                }
            )
    suffix = "" if max_models is None and model_filter is None else "_smoke"
    write_csv(OUTDIR / f"matbench_discovery_fiber_sensitivity{suffix}.csv", rows)
    write_csv(
        OUTDIR / f"matbench_discovery_fiber_sensitivity_summary{suffix}.csv",
        summarize_by_method(rows),
    )


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


def jarvis_sensitivity() -> None:
    paths = github_tree_paths()
    formation = contribution_map(paths, "formation_energy_peratom")
    bandgap = contribution_map(paths, "optb88vdw_bandgap")
    shared = sorted(set(formation) & set(bandgap))
    bandgap_ref = read_zipped_json(
        "jarvis_leaderboard/benchmarks/AI/SinglePropertyPrediction/"
        "dft_3d_optb88vdw_bandgap.json.zip"
    )["test"]
    formation_ref = read_zipped_json(
        "jarvis_leaderboard/benchmarks/AI/SinglePropertyPrediction/"
        "dft_3d_formation_energy_peratom.json.zip"
    )["test"]

    rows: list[dict[str, object]] = []
    for model_id in shared:
        try:
            formation_pred = read_zipped_predictions(formation[model_id])
            ids = sorted(set(formation_pred) & set(bandgap_ref) & set(formation_ref))
            if len(ids) < 50:
                continue
            pred = np.array([formation_pred[i] for i in ids], dtype=float)
            truth = np.array([formation_ref[i] for i in ids], dtype=float)
            labels = np.array([bandgap_ref[i] > 1.0 for i in ids], dtype=bool)
            abs_err = np.abs(pred - truth)
            for n_bins in [5, 10, 20, 40]:
                ambiguous, total, cert, median_size = quantile_fiber_stats(pred, labels, n_bins, min_size=5)
                rows.append(
                    {
                        "source": "jarvis_leaderboard",
                        "method": f"quantile_{n_bins}",
                        "model_id": model_id,
                        "n_candidates": len(ids),
                        "ambiguous_fibers": ambiguous,
                        "total_fibers": total,
                        "ambiguous_fraction": ambiguous / total if total else math.nan,
                        "certifiable_fraction": cert / len(ids),
                        "median_fiber_size": median_size,
                        "tolerance": math.nan,
                    }
                )
            for k in [25, 50, 100]:
                cert_frac, median_size = knn_certifiable_fraction(pred, labels, k)
                rows.append(
                    {
                        "source": "jarvis_leaderboard",
                        "method": f"knn_{k}",
                        "model_id": model_id,
                        "n_candidates": len(ids),
                        "ambiguous_fibers": math.nan,
                        "total_fibers": math.nan,
                        "ambiguous_fraction": math.nan,
                        "certifiable_fraction": cert_frac,
                        "median_fiber_size": median_size,
                        "tolerance": math.nan,
                    }
                )
            for method, tolerance in [
                ("interval_mae", float(abs_err.mean())),
                ("interval_q80_abs_error", float(np.quantile(abs_err, 0.80))),
            ]:
                cert_frac, median_size = interval_fiber_stats(pred, labels, tolerance)
                rows.append(
                    {
                        "source": "jarvis_leaderboard",
                        "method": method,
                        "model_id": model_id,
                        "n_candidates": len(ids),
                        "ambiguous_fibers": math.nan,
                        "total_fibers": math.nan,
                        "ambiguous_fraction": math.nan,
                        "certifiable_fraction": cert_frac,
                        "median_fiber_size": median_size,
                        "tolerance": tolerance,
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    "source": "jarvis_leaderboard",
                    "method": "error",
                    "model_id": model_id,
                    "n_candidates": 0,
                    "ambiguous_fibers": 0,
                    "total_fibers": 0,
                    "ambiguous_fraction": math.nan,
                    "certifiable_fraction": math.nan,
                    "median_fiber_size": math.nan,
                    "tolerance": math.nan,
                    "error": str(exc),
                }
            )
    write_csv(OUTDIR / "jarvis_fiber_sensitivity.csv", rows)
    write_csv(OUTDIR / "jarvis_fiber_sensitivity_summary.csv", summarize_by_method(rows))


def summarize_by_method(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    clean = [row for row in rows if row.get("method") != "error"]
    methods = sorted({str(row["method"]) for row in clean})
    out: list[dict[str, object]] = []
    for method in methods:
        sub = [row for row in clean if row["method"] == method]
        cert = np.array([float(row["certifiable_fraction"]) for row in sub], dtype=float)
        amb_vals = [
            float(row["ambiguous_fraction"])
            for row in sub
            if not pd.isna(row.get("ambiguous_fraction", math.nan))
        ]
        fiber_sizes = [
            float(row["median_fiber_size"])
            for row in sub
            if not pd.isna(row.get("median_fiber_size", math.nan))
        ]
        tolerances = [
            float(row["tolerance"])
            for row in sub
            if not pd.isna(row.get("tolerance", math.nan))
        ]
        out.append(
            {
                "method": method,
                "models": len(sub),
                "median_certifiable_fraction": float(np.median(cert)) if len(cert) else math.nan,
                "mean_certifiable_fraction": float(np.mean(cert)) if len(cert) else math.nan,
                "median_ambiguous_fraction": float(np.median(amb_vals)) if amb_vals else math.nan,
                "median_fiber_size": float(np.median(fiber_sizes)) if fiber_sizes else math.nan,
                "median_tolerance": float(np.median(tolerances)) if tolerances else math.nan,
            }
        )
    return out


def main() -> None:
    conformal_rows = [conformal_once(seed) for seed in range(50)]
    write_csv(OUTDIR / "revision_conformal_repeated_runs.csv", conformal_rows)
    write_csv(
        OUTDIR / "revision_conformal_repeated_summary.csv",
        summarize_numeric(
            conformal_rows,
            [
                "deployment_residual",
                "benchmark_conformal_coverage",
                "benchmark_transfer_coverage",
                "oracle_deployment_coverage",
                "response_rank_coverage",
                "benchmark_width",
                "oracle_width",
                "response_rank_width",
            ],
        ),
    )
    leaderboard_repeats()
    matbench_sensitivity()
    jarvis_sensitivity()


if __name__ == "__main__":
    main()
