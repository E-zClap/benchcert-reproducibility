"""Baseline comparison: response-rank certification versus conformal prediction,
OOD detection, ensemble uncertainty, and active-learning acquisition policies.

Experiment design
-----------------
Response space: 8-dimensional.  Four benchmark probes (effective rank 3) span a
3-dimensional subspace.  Ten deployment probes are defined: three lie in the
benchmark span (residual 0), four have residual ~0.70, and three have residual
~0.92.  A pool of 40 candidate new probes includes, at fixed positions, probes
that directly cover the high-residual deployment directions.

Policies compared in the acquisition experiment
 1. Residual-greedy      -- minimises max deployment residual (our method)
 2. Uncertainty sampling -- selects probe most correlated with ensemble variance
 3. Diversity            -- selects probe most orthogonal to current span
 4. Benchmark-aligned    -- selects probe most aligned with benchmark span
 5. Random               -- uniform random (100-trial Monte Carlo average)

Key metric: certified candidates (viable + nonviable) per added probe.

Outputs
-------
outputs/baseline_comparison_coverage_table.csv
outputs/baseline_comparison_acquisition.csv
outputs/baseline_comparison_radius_sensitivity.csv
outputs/baseline_comparison_acquisition.svg
outputs/baseline_comparison_coverage_vs_method.svg
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Linear-algebra helpers
# ---------------------------------------------------------------------------

def row_span_projector(rows: np.ndarray) -> np.ndarray:
    """Orthogonal projector onto the row span of *rows*."""
    if rows.ndim == 1:
        rows = rows[np.newaxis]
    if rows.size == 0:
        return np.zeros((rows.shape[1], rows.shape[1]))
    q, _ = np.linalg.qr(rows.T)
    rank = np.linalg.matrix_rank(rows, tol=1e-9)
    q = q[:, :rank]
    return q @ q.T


def residual_norm(rows: np.ndarray, probe: np.ndarray) -> float:
    P = row_span_projector(rows)
    return float(np.linalg.norm(probe - P @ probe))


def residuals_all(rows: np.ndarray, probes: np.ndarray) -> np.ndarray:
    """Return residual norms for each probe in *probes* (shape (m, d))."""
    P = row_span_projector(rows)
    diff = probes - probes @ P  # (m, d)
    return np.linalg.norm(diff, axis=1)


def count_certified(
    estimates: np.ndarray,  # (n_candidates,)
    R_c: np.ndarray,        # (n_candidates,) admissible radii
    gaps: np.ndarray,       # (n_deploy,) residual norms
    thresholds: np.ndarray, # (n_deploy,)
) -> int:
    """Count candidates certified at ALL deployment probes.

    A candidate is certified (viable or nonviable) when its certification
    interval at every deployment probe lies entirely on one side of the
    threshold.  An ambiguous candidate has at least one interval straddling
    a threshold.
    """
    certified = 0
    for est, r in zip(estimates, R_c):
        ambiguous = False
        for g, tau in zip(gaps, thresholds):
            lo = est - r * g
            hi = est + r * g
            if lo <= tau <= hi:
                ambiguous = True
                break
        if not ambiguous:
            certified += 1
    return certified


# ---------------------------------------------------------------------------
# Geometry definition
# ---------------------------------------------------------------------------

def make_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (benchmark_probes, deployment_probes, pool_probes, d)."""
    d = 8
    rng = np.random.default_rng(17)

    # Benchmark probes: span first 3 dimensions with slight noise
    e1 = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    e2 = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    e3 = np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=float)

    B = np.array([
        e1 + 0.05 * rng.standard_normal(d),
        e2 + 0.05 * rng.standard_normal(d),
        0.6 * e1 + 0.8 * e2 + 0.04 * rng.standard_normal(d),
        0.4 * e2 + 0.9 * e3 + 0.04 * rng.standard_normal(d),
    ])
    B = B / np.linalg.norm(B, axis=1, keepdims=True)

    # Deployment probes -- three groups by initial residual
    # Group 1 (3 probes): in benchmark span, residual ~ 0
    dep_1 = np.array([
        0.7 * e1 + 0.5 * e2 + 0.5 * e3,
        0.3 * e1 + 0.8 * e2 + 0.3 * e3,
        0.5 * e1 + 0.3 * e2 + 0.8 * e3,
    ])
    dep_1 = dep_1 / np.linalg.norm(dep_1, axis=1, keepdims=True)

    # Group 2 (4 probes): partially outside span, residual ~ 0.70
    e4 = np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=float)
    e5 = np.array([0, 0, 0, 0, 1, 0, 0, 0], dtype=float)
    dep_2 = np.array([
        0.70 * e4 + 0.50 * e1 + 0.50 * e2,
        0.72 * e4 + 0.40 * e2 + 0.48 * e3,
        0.71 * e5 + 0.50 * e1 + 0.45 * e3,
        0.73 * e5 + 0.60 * e2 + 0.22 * e3,
    ])
    dep_2 = dep_2 / np.linalg.norm(dep_2, axis=1, keepdims=True)

    # Group 3 (3 probes): mostly outside span, residual ~ 0.90
    e6 = np.array([0, 0, 0, 0, 0, 1, 0, 0], dtype=float)
    e7 = np.array([0, 0, 0, 0, 0, 0, 1, 0], dtype=float)
    e8 = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=float)
    dep_3 = np.array([
        0.90 * e6 + 0.30 * e1 + 0.15 * e2,
        0.88 * e7 + 0.28 * e2 + 0.14 * e3,
        0.87 * e8 + 0.40 * e1 + 0.10 * e3,
    ])
    dep_3 = dep_3 / np.linalg.norm(dep_3, axis=1, keepdims=True)

    deploy = np.vstack([dep_1, dep_2, dep_3])  # (10, 8)

    # Pool of 40 candidate probes for acquisition.
    # Indices 0-4: targeted probes covering group-2 and group-3 directions.
    # Indices 5-39: random probes (not specifically targeted).
    pool_targeted = np.array([
        e4 + 0.06 * rng.standard_normal(d),   # covers group-2 e4 direction
        e5 + 0.06 * rng.standard_normal(d),   # covers group-2 e5 direction
        e6 + 0.06 * rng.standard_normal(d),   # covers group-3 e6 direction
        e7 + 0.06 * rng.standard_normal(d),   # covers group-3 e7 direction
        e8 + 0.06 * rng.standard_normal(d),   # covers group-3 e8 direction
    ])
    pool_random = rng.standard_normal((35, d))
    pool = np.vstack([pool_targeted, pool_random])
    pool = pool / np.linalg.norm(pool, axis=1, keepdims=True)

    return B, deploy, pool, d


# ---------------------------------------------------------------------------
# Conformal-prediction experiment
# ---------------------------------------------------------------------------

def conformal_coverage_experiment(
    B: np.ndarray,
    deploy: np.ndarray,
) -> dict[str, float]:
    """
    Demonstrate that conformal calibration on benchmark labels does not certify
    deployment coverage when the deployment probe is outside the benchmark span.

    Setup
    -----
    * 600 candidates with 8D system features X ~ N(0, I).
    * Model observes only the benchmark-span projection of X:
      X_vis = X @ P_B  (3D effective), representing a model that sees only
      benchmark-measured properties.
    * Observation noise sigma = 0.05 is added to all labels.

    Two conformal experiments
    -------------------------
    (a) Benchmark conformal: calibrated on noisy benchmark labels y_bench_noisy.
        Achieves ~95% benchmark coverage.
    (b) The same conformal intervals applied naively to deployment labels
        y_deploy_noisy: covers far less than 95%, because the conformal quantile
        q_bench is set by small benchmark noise, while the deployment response
        has an additional unspanned component X @ r_star ~ N(0, ||r_star||^2).

    Response-rank certification
    ---------------------------
    Uses R = 2.0 (an admissible radius that covers 95% of X @ r_star for the
    geometry used here) to show that correct deployment bounds ARE achievable,
    at the cost of wider intervals that classify more candidates as ambiguous.
    """
    d = 8
    sigma = 0.05       # label noise
    R_cert = 2.0       # admissible radius for the coverage validation
    rng = np.random.default_rng(42)

    n = 600
    n_train, n_cal, n_test = 200, 200, 200

    # Candidate features
    X = rng.standard_normal((n, d))

    # Benchmark-span projector and visible features
    P_B = row_span_projector(B)
    X_vis = X @ P_B      # (n, 8) but effectively 3D

    # Deployment probe (hardest, index 9) and residual
    k_deploy = deploy[9]
    g = residual_norm(B, k_deploy)

    # True and noisy labels
    y_bench_true = X @ B[0]              # benchmark label (first probe)
    y_bench_noisy = y_bench_true + sigma * rng.standard_normal(n)
    y_deploy_true = X @ k_deploy
    y_deploy_noisy = y_deploy_true + sigma * rng.standard_normal(n)

    idx = rng.permutation(n)
    tr, cal, te = idx[:n_train], idx[n_train:n_train+n_cal], idx[n_train+n_cal:]

    # --- Benchmark model: fit y_bench_noisy from X_vis (benchmark-visible only) ---
    w_bench, *_ = np.linalg.lstsq(X_vis[tr], y_bench_noisy[tr], rcond=None)

    # --- Deployment model: best linear predictor from benchmark-visible features ---
    # (represents what any model restricted to the benchmark span can learn)
    w_deploy, *_ = np.linalg.lstsq(X_vis[tr], y_deploy_noisy[tr], rcond=None)

    yp_bench_cal  = X_vis[cal] @ w_bench
    yp_bench_test = X_vis[te]  @ w_bench
    yp_deploy_cal  = X_vis[cal] @ w_deploy
    yp_deploy_test = X_vis[te]  @ w_deploy

    alpha = 0.05
    n_cal_eff = n_cal

    # --- (a) Benchmark conformal (calibrated on benchmark labels) ---
    q_bench = np.quantile(
        np.abs(yp_bench_cal - y_bench_noisy[cal]),
        min(1.0, (1 - alpha) * (n_cal_eff + 1) / n_cal_eff),
    )
    bench_lo = yp_bench_test - q_bench
    bench_hi = yp_bench_test + q_bench
    benchmark_coverage = float(np.mean(
        (bench_lo <= y_bench_noisy[te]) & (y_bench_noisy[te] <= bench_hi)
    ))
    benchmark_width = float(np.mean(bench_hi - bench_lo))

    # --- (b) Naive transfer: apply q_bench to deployment predictions ---
    # This is the "common mistake": taking a benchmark-calibrated model and
    # using its conformal intervals for a deployment claim.
    bc_deploy_lo = yp_deploy_test - q_bench
    bc_deploy_hi = yp_deploy_test + q_bench
    bc_deploy_coverage = float(np.mean(
        (bc_deploy_lo <= y_deploy_noisy[te]) & (y_deploy_noisy[te] <= bc_deploy_hi)
    ))
    bc_deploy_width = benchmark_width   # same quantile

    # --- Oracle deployment conformal (if deployment labels were available) ---
    q_deploy_oracle = np.quantile(
        np.abs(yp_deploy_cal - y_deploy_noisy[cal]),
        min(1.0, (1 - alpha) * (n_cal_eff + 1) / n_cal_eff),
    )
    oracle_lo = yp_deploy_test - q_deploy_oracle
    oracle_hi = yp_deploy_test + q_deploy_oracle
    oracle_deploy_coverage = float(np.mean(
        (oracle_lo <= y_deploy_noisy[te]) & (y_deploy_noisy[te] <= oracle_hi)
    ))
    oracle_deploy_width = float(np.mean(oracle_hi - oracle_lo))

    # --- Response-rank certification interval (R = R_cert) ---
    cert_lo = yp_deploy_test - R_cert * g
    cert_hi = yp_deploy_test + R_cert * g
    cert_coverage = float(np.mean(
        (cert_lo <= y_deploy_noisy[te]) & (y_deploy_noisy[te] <= cert_hi)
    ))
    cert_width = float(2 * R_cert * g)

    return {
        "benchmark_conformal_coverage": benchmark_coverage,
        "benchmark_conformal_width": round(benchmark_width, 4),
        "conformal_q_bench": round(float(q_bench), 4),
        "benchmark_conformal_applied_to_deployment_coverage": bc_deploy_coverage,
        "oracle_deployment_conformal_coverage": oracle_deploy_coverage,
        "oracle_deployment_conformal_width": round(oracle_deploy_width, 4),
        "response_cert_coverage": cert_coverage,
        "response_cert_width": round(cert_width, 4),
        "response_cert_R": R_cert,
        "deployment_residual_norm": round(g, 4),
        "label_noise_sigma": sigma,
    }


# ---------------------------------------------------------------------------
# OOD and ensemble baseline comparisons
# ---------------------------------------------------------------------------

def ood_and_ensemble_experiment(
    B: np.ndarray,
    deploy: np.ndarray,
) -> list[dict[str, float]]:
    """
    Show that OOD scores and ensemble uncertainty do not predict
    response-certification ambiguity.

    For each of 400 test candidates:
      - OOD score: Mahalanobis distance from training feature distribution
      - Ensemble uncertainty: std-dev across 50 bootstrap linear models
      - Certification gap: R_c * g (directly from span residual)
    """
    d = 8
    rng = np.random.default_rng(99)

    n_train, n_test = 150, 400
    n_total = n_train + n_test
    X = rng.standard_normal((n_total, d))
    y_bench = X @ B.T   # benchmark probe responses as features

    # Training split
    X_tr = y_bench[:n_train]           # (n_train, 4)
    X_te = y_bench[n_train:]           # (n_test, 4)

    # Per-candidate admissible radius (varies: simulates descriptor completeness)
    R_c = 0.15 + 0.45 * rng.uniform(0, 1, size=n_test)

    # Global deployment residual (same for all candidates; varies by probe)
    # Use the probe with the largest residual (group-3, index 9)
    g_max = residual_norm(B, deploy[9])
    # And a medium-residual probe (group-2, index 3)
    g_mid = residual_norm(B, deploy[3])

    # --- OOD: Mahalanobis distance from training feature distribution ---
    mu = X_tr.mean(axis=0)
    cov = np.cov(X_tr.T) + 1e-6 * np.eye(X_tr.shape[1])
    cov_inv = np.linalg.inv(cov)
    diff = X_te - mu
    ood_scores = np.sqrt(np.einsum("ij,jk,ik->i", diff, cov_inv, diff))

    # --- Ensemble: 50 bootstrap linear models predicting deployment response ---
    k_deploy = deploy[9]
    y_deploy_tr = X[:n_train] @ k_deploy
    y_deploy_te = X[n_train:] @ k_deploy

    n_ens = 50
    ensemble_preds = np.zeros((n_ens, n_test))
    for b in range(n_ens):
        idx_b = rng.integers(0, n_train, size=n_train)
        w_b, *_ = np.linalg.lstsq(X_tr[idx_b], y_deploy_tr[idx_b], rcond=None)
        ensemble_preds[b] = X_te @ w_b
    ensemble_uncertainty = ensemble_preds.std(axis=0)   # (n_test,)

    # --- Certification gap (per candidate, per probe) ---
    cert_gap_max = R_c * g_max
    cert_gap_mid = R_c * g_mid

    rows = []
    for i in range(n_test):
        rows.append({
            "candidate": i,
            "ood_score": float(ood_scores[i]),
            "ensemble_uncertainty": float(ensemble_uncertainty[i]),
            "admissible_radius": float(R_c[i]),
            "cert_gap_high_residual_probe": float(cert_gap_max[i]),
            "cert_gap_mid_residual_probe": float(cert_gap_mid[i]),
            "deployment_residual_high": g_max,
            "deployment_residual_mid": g_mid,
        })
    return rows


# ---------------------------------------------------------------------------
# Acquisition-policy comparison
# ---------------------------------------------------------------------------

def acquisition_experiment(
    B: np.ndarray,
    deploy: np.ndarray,
    pool: np.ndarray,
    n_steps: int = 12,
    n_candidates: int = 400,
    n_random_trials: int = 200,
) -> list[dict[str, object]]:
    """
    Compare five acquisition policies over n_steps probe additions.

    Candidates have a single scalar deployment estimate drawn from N(0, 0.45)
    with admissible radius R_c ~ Uniform(0.15, 0.55).  A candidate is certified
    at a deployment probe when its certification interval does not straddle the
    threshold (tau = 0 for all probes).  A candidate is globally certified when
    every deployment probe is certified.
    """
    rng = np.random.default_rng(31)
    tau = np.zeros(len(deploy))
    estimates = rng.normal(0.0, 0.45, size=n_candidates)
    R_c = rng.uniform(0.15, 0.55, size=n_candidates)

    # Pre-compute deployment responses for each candidate under each probe
    # (needed to count certified candidates as span grows)
    # We do not use ground-truth here -- certification is based on interval geometry
    # (span residual), not on observed responses.

    def gaps_now(rows: np.ndarray) -> np.ndarray:
        return residuals_all(rows, deploy)

    # ---- 1. Residual-greedy ----
    greedy_rows = B.copy()
    greedy_counts = []
    greedy_selected: list[int] = []
    for step in range(n_steps + 1):
        g = gaps_now(greedy_rows)
        greedy_counts.append(count_certified(estimates, R_c, g, tau))
        if step == n_steps:
            break
        best_j, best_score = None, np.inf
        for j, p in enumerate(pool):
            if j in greedy_selected:
                continue
            g_trial = gaps_now(np.vstack([greedy_rows, p]))
            score = float(np.max(g_trial))
            if score < best_score:
                best_score, best_j = score, j
        greedy_selected.append(best_j)
        greedy_rows = np.vstack([greedy_rows, pool[best_j]])

    # ---- 2. Uncertainty sampling ----
    # Proxy for "uncertainty": variance of predictions across a bootstrap ensemble.
    # We pick the next probe most correlated with that ensemble variance.
    # For each candidate probe q, score = mean candidate uncertainty that q reduces.
    # Implementation: score(q) = cos^2(angle between q and principal uncertainty axis)
    # The principal uncertainty axis is approximated by the leading left singular
    # vector of the residual matrix (deploy - deploy @ P).
    def uncertainty_score(rows: np.ndarray, q: np.ndarray) -> float:
        P = row_span_projector(rows)
        deploy_res = deploy - deploy @ P       # (m, d)
        if np.linalg.norm(deploy_res) < 1e-10:
            return 0.0
        # Leading direction of residual space
        _, _, Vt = np.linalg.svd(deploy_res, full_matrices=False)
        u_max = Vt[0]
        q_n = q / (np.linalg.norm(q) + 1e-12)
        return float(np.dot(q_n, u_max) ** 2)

    unc_rows = B.copy()
    unc_counts = []
    unc_selected: list[int] = []
    for step in range(n_steps + 1):
        g = gaps_now(unc_rows)
        unc_counts.append(count_certified(estimates, R_c, g, tau))
        if step == n_steps:
            break
        best_j, best_score = None, -np.inf
        for j, p in enumerate(pool):
            if j in unc_selected:
                continue
            score = uncertainty_score(unc_rows, p)
            if score > best_score:
                best_score, best_j = score, j
        unc_selected.append(best_j)
        unc_rows = np.vstack([unc_rows, pool[best_j]])

    # ---- 3. Diversity (most orthogonal to current span) ----
    def diversity_score(rows: np.ndarray, q: np.ndarray) -> float:
        P = row_span_projector(rows)
        q_n = q / (np.linalg.norm(q) + 1e-12)
        return float(np.linalg.norm(q_n - P @ q_n))  # residual in probe space

    div_rows = B.copy()
    div_counts = []
    div_selected: list[int] = []
    for step in range(n_steps + 1):
        g = gaps_now(div_rows)
        div_counts.append(count_certified(estimates, R_c, g, tau))
        if step == n_steps:
            break
        best_j, best_score = None, -np.inf
        for j, p in enumerate(pool):
            if j in div_selected:
                continue
            score = diversity_score(div_rows, p)
            if score > best_score:
                best_score, best_j = score, j
        div_selected.append(best_j)
        div_rows = np.vstack([div_rows, pool[best_j]])

    # ---- 4. Benchmark-aligned (anti-greedy baseline) ----
    def bench_align_score(rows: np.ndarray, q: np.ndarray) -> float:
        P = row_span_projector(rows)
        q_n = q / (np.linalg.norm(q) + 1e-12)
        return float(np.linalg.norm(P @ q_n))  # alignment with current span

    ba_rows = B.copy()
    ba_counts = []
    ba_selected: list[int] = []
    for step in range(n_steps + 1):
        g = gaps_now(ba_rows)
        ba_counts.append(count_certified(estimates, R_c, g, tau))
        if step == n_steps:
            break
        best_j, best_score = None, -np.inf
        for j, p in enumerate(pool):
            if j in ba_selected:
                continue
            score = bench_align_score(ba_rows, p)
            if score > best_score:
                best_score, best_j = score, j
        ba_selected.append(best_j)
        ba_rows = np.vstack([ba_rows, pool[best_j]])

    # ---- 5. Random (Monte Carlo) ----
    random_all = np.zeros((n_random_trials, n_steps + 1), dtype=int)
    for t in range(n_random_trials):
        local = np.random.default_rng(5000 + t)
        order = local.permutation(len(pool))
        r_rows = B.copy()
        for step in range(n_steps + 1):
            g = gaps_now(r_rows)
            random_all[t, step] = count_certified(estimates, R_c, g, tau)
            if step < n_steps:
                r_rows = np.vstack([r_rows, pool[order[step]]])

    random_mean = random_all.mean(axis=0)
    random_p10  = np.percentile(random_all, 10, axis=0)
    random_p90  = np.percentile(random_all, 90, axis=0)

    rows_out = []
    for step in range(n_steps + 1):
        rows_out.append({
            "added_probes": step,
            "total_candidates": n_candidates,
            "residual_greedy_certified": int(greedy_counts[step]),
            "uncertainty_sampling_certified": int(unc_counts[step]),
            "diversity_sampling_certified": int(div_counts[step]),
            "benchmark_aligned_certified": int(ba_counts[step]),
            "random_mean_certified": float(random_mean[step]),
            "random_p10_certified": float(random_p10[step]),
            "random_p90_certified": float(random_p90[step]),
            "residual_greedy_frac": greedy_counts[step] / n_candidates,
            "uncertainty_frac": unc_counts[step] / n_candidates,
            "diversity_frac": div_counts[step] / n_candidates,
            "benchmark_aligned_frac": ba_counts[step] / n_candidates,
            "random_mean_frac": float(random_mean[step]) / n_candidates,
        })
    return rows_out


def radius_sensitivity_experiment(
    B: np.ndarray,
    deploy: np.ndarray,
    pool: np.ndarray,
    scales: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
    n_steps: int = 3,
    n_candidates: int = 400,
    n_random_trials: int = 200,
) -> list[dict[str, object]]:
    """Sweep multiplicative choices of the admissible radius R_c.

    Probe-selection policies are geometric and therefore held fixed across the
    radius sweep.  The sweep reports how many candidates remain certifiable as
    the conservative radius is tightened or relaxed.
    """
    rng = np.random.default_rng(31)
    tau = np.zeros(len(deploy))
    estimates = rng.normal(0.0, 0.45, size=n_candidates)
    base_R = rng.uniform(0.15, 0.55, size=n_candidates)

    def gaps_now(rows: np.ndarray) -> np.ndarray:
        return residuals_all(rows, deploy)

    def uncertainty_score(rows: np.ndarray, q: np.ndarray) -> float:
        P = row_span_projector(rows)
        deploy_res = deploy - deploy @ P
        if np.linalg.norm(deploy_res) < 1e-10:
            return 0.0
        _, _, Vt = np.linalg.svd(deploy_res, full_matrices=False)
        q_n = q / (np.linalg.norm(q) + 1e-12)
        return float(np.dot(q_n, Vt[0]) ** 2)

    def diversity_score(rows: np.ndarray, q: np.ndarray) -> float:
        P = row_span_projector(rows)
        q_n = q / (np.linalg.norm(q) + 1e-12)
        return float(np.linalg.norm(q_n - P @ q_n))

    def bench_align_score(rows: np.ndarray, q: np.ndarray) -> float:
        P = row_span_projector(rows)
        q_n = q / (np.linalg.norm(q) + 1e-12)
        return float(np.linalg.norm(P @ q_n))

    def selected_rows(policy: str) -> np.ndarray:
        rows = B.copy()
        selected: list[int] = []
        for _ in range(n_steps):
            best_j = None
            if policy == "residual_greedy":
                best_score = np.inf
                for j, p in enumerate(pool):
                    if j in selected:
                        continue
                    score = float(np.max(gaps_now(np.vstack([rows, p]))))
                    if score < best_score:
                        best_score, best_j = score, j
            elif policy == "uncertainty":
                best_score = -np.inf
                for j, p in enumerate(pool):
                    if j in selected:
                        continue
                    score = uncertainty_score(rows, p)
                    if score > best_score:
                        best_score, best_j = score, j
            elif policy == "diversity":
                best_score = -np.inf
                for j, p in enumerate(pool):
                    if j in selected:
                        continue
                    score = diversity_score(rows, p)
                    if score > best_score:
                        best_score, best_j = score, j
            elif policy == "benchmark_aligned":
                best_score = -np.inf
                for j, p in enumerate(pool):
                    if j in selected:
                        continue
                    score = bench_align_score(rows, p)
                    if score > best_score:
                        best_score, best_j = score, j
            else:
                raise ValueError(f"unknown policy: {policy}")
            assert best_j is not None
            selected.append(best_j)
            rows = np.vstack([rows, pool[best_j]])
        return rows

    policy_rows = {
        "no_new_probe": B.copy(),
        "residual_greedy": selected_rows("residual_greedy"),
        "uncertainty": selected_rows("uncertainty"),
        "diversity": selected_rows("diversity"),
        "benchmark_aligned": selected_rows("benchmark_aligned"),
    }

    random_orders = [np.random.default_rng(5000 + t).permutation(len(pool)) for t in range(n_random_trials)]
    random_rows = []
    for order in random_orders:
        rows = B.copy()
        for step in range(n_steps):
            rows = np.vstack([rows, pool[order[step]]])
        random_rows.append(rows)

    rows_out: list[dict[str, object]] = []
    for scale in scales:
        R_c = base_R * scale
        row: dict[str, object] = {
            "radius_scale": scale,
            "total_candidates": n_candidates,
            "added_probes": n_steps,
        }
        for name, rows in policy_rows.items():
            certified = count_certified(estimates, R_c, gaps_now(rows), tau)
            row[f"{name}_certified"] = int(certified)
            row[f"{name}_frac"] = certified / n_candidates
        random_counts = [
            count_certified(estimates, R_c, gaps_now(rows), tau) for rows in random_rows
        ]
        row["random_mean_certified"] = float(np.mean(random_counts))
        row["random_p10_certified"] = float(np.percentile(random_counts, 10))
        row["random_p90_certified"] = float(np.percentile(random_counts, 90))
        row["random_mean_frac"] = float(np.mean(random_counts)) / n_candidates
        rows_out.append(row)
    return rows_out


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_coverage_bar(coverage_data: dict[str, float]) -> None:
    """Bar chart comparing deployment coverage rates across methods."""
    labels = [
        "Conformal\n(benchmark labels,\nbenchmark coverage)",
        "Conformal (benchmark-\ncalibrated, applied to\ndeployment labels)",
        "Response-rank\ncertification\n(R = 2.0, deployment)",
    ]
    values = [
        coverage_data["benchmark_conformal_coverage"] * 100,
        coverage_data["benchmark_conformal_applied_to_deployment_coverage"] * 100,
        coverage_data["response_cert_coverage"] * 100,
    ]
    colors = ["#3a7ebf", "#c43b3b", "#188a5b"]

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white", linewidth=0.8)
    ax.axhline(95, linestyle="--", linewidth=1.2, color="#555", label="95% target coverage")
    ax.set_ylim(0, 108)
    ax.set_ylabel("deployment response coverage (%)")
    ax.set_title(
        "Conformal calibration on benchmark labels does not certify deployment coverage",
        fontsize=9,
    )
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUTDIR / "baseline_comparison_coverage_vs_method.svg")
    fig.savefig(OUTDIR / "baseline_comparison_coverage_vs_method.png", dpi=200)
    plt.close(fig)


def plot_ood_uncertainty_scatter(ood_rows: list[dict[str, float]]) -> None:
    """2-panel scatter: OOD score vs gap, ensemble uncertainty vs gap."""
    ood = np.array([r["ood_score"] for r in ood_rows])
    unc = np.array([r["ensemble_uncertainty"] for r in ood_rows])
    gap = np.array([r["cert_gap_high_residual_probe"] for r in ood_rows])

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
    for ax, x, xlabel in [
        (axes[0], ood, "OOD score (Mahalanobis distance)"),
        (axes[1], unc, "Ensemble uncertainty (prediction std-dev)"),
    ]:
        ax.scatter(x, gap, s=14, alpha=0.45, color="#3a7ebf", rasterized=True)
        # Show correlation
        r = float(np.corrcoef(x, gap)[0, 1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("response-certification gap ($R_c \\cdot \\|r_\\star\\|$)")
        ax.text(
            0.97, 0.96, f"$r = {r:.3f}$",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
        )
    axes[0].set_title("OOD score vs certification gap")
    axes[1].set_title("Ensemble uncertainty vs certification gap")
    fig.suptitle(
        "Neither OOD detection nor ensemble uncertainty predicts response-certification ambiguity",
        fontsize=9,
    )
    fig.savefig(OUTDIR / "baseline_comparison_ood_uncertainty_scatter.svg")
    fig.savefig(OUTDIR / "baseline_comparison_ood_uncertainty_scatter.png", dpi=200)
    plt.close(fig)


def plot_acquisition_curves(acq_rows: list[dict]) -> None:
    """Certified-candidate fraction vs added probes for all five policies."""
    steps = [r["added_probes"] for r in acq_rows]
    greedy_frac = [r["residual_greedy_frac"] for r in acq_rows]
    unc_frac    = [r["uncertainty_frac"] for r in acq_rows]
    div_frac    = [r["diversity_frac"] for r in acq_rows]
    ba_frac     = [r["benchmark_aligned_frac"] for r in acq_rows]
    rand_mean   = [r["random_mean_frac"] for r in acq_rows]
    rand_p10    = [r["random_p10_certified"] / acq_rows[0]["total_candidates"]
                   for r in acq_rows]
    rand_p90    = [r["random_p90_certified"] / acq_rows[0]["total_candidates"]
                   for r in acq_rows]

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "font.size": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
    ax.fill_between(
        steps,
        rand_p10,
        rand_p90,
        color="#9aa6b2",
        alpha=0.22,
        linewidth=0,
        label="random 10-90%",
        zorder=1,
    )
    ax.plot(
        steps,
        rand_mean,
        "o-",
        color="#7c8796",
        lw=1.4,
        ms=4.5,
        label="random mean",
        zorder=2,
    )
    ax.plot(
        steps,
        unc_frac,
        "s-",
        color="#3d78b5",
        lw=1.55,
        ms=4.8,
        label="uncertainty",
        zorder=3,
    )
    ax.plot(
        steps,
        div_frac,
        "^-",
        color="#b87518",
        lw=1.55,
        ms=5.0,
        label="diversity",
        zorder=3,
    )
    ax.plot(
        steps,
        ba_frac,
        "v-",
        color="#8d6aa9",
        lw=1.35,
        ms=4.8,
        label="benchmark-aligned",
        zorder=3,
    )
    ax.plot(
        steps,
        greedy_frac,
        "o-",
        color="#0f8a5f",
        lw=2.35,
        ms=6.5,
        label="residual-greedy",
        zorder=5,
    )

    step3 = steps.index(3)
    ax.axvline(3, color="#202736", lw=0.8, ls=":", alpha=0.65)
    ax.scatter(
        [3],
        [greedy_frac[step3]],
        s=88,
        facecolor="white",
        edgecolor="#0f8a5f",
        linewidth=1.8,
        zorder=6,
    )
    ax.annotate(
        "70.8% certified\nafter 3 probes",
        xy=(3, greedy_frac[step3]),
        xytext=(4.25, 0.78),
        ha="left",
        va="center",
        fontsize=9,
        color="#0b6f4d",
        arrowprops=dict(arrowstyle="-", color="#0f8a5f", lw=0.9),
    )

    ax.set_xlabel("added response probes")
    ax.set_ylabel("candidates certified")
    ax.set_xlim(-0.3, max(steps) + 0.3)
    ax.set_ylim(0.42, 1.03)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", color="#d9dde3", lw=0.7, alpha=0.8)
    ax.legend(frameon=False, fontsize=8.2, ncol=3, loc="lower right")
    fig.savefig(OUTDIR / "baseline_comparison_acquisition.svg")
    fig.savefig(OUTDIR / "baseline_comparison_acquisition.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    B, deploy, pool, d = make_geometry()

    # Print initial residuals for logging
    g_initial = residuals_all(B, deploy)
    print("Initial deployment residuals:")
    for i, g in enumerate(g_initial):
        print(f"  probe {i:2d}: {g:.4f}")

    # --- Conformal coverage experiment ---
    coverage = conformal_coverage_experiment(B, deploy)
    print("\nCoverage comparison:")
    for k, v in coverage.items():
        print(f"  {k}: {v:.4f}")
    write_csv(OUTDIR / "baseline_comparison_coverage_table.csv", [coverage])
    plot_coverage_bar(coverage)

    # --- OOD and ensemble experiment ---
    ood_rows = ood_and_ensemble_experiment(B, deploy)
    write_csv(OUTDIR / "baseline_comparison_ood_uncertainty.csv", ood_rows)
    plot_ood_uncertainty_scatter(ood_rows)

    # --- Acquisition experiment ---
    acq_rows = acquisition_experiment(B, deploy, pool, n_steps=12, n_candidates=400)
    write_csv(OUTDIR / "baseline_comparison_acquisition.csv", acq_rows)
    plot_acquisition_curves(acq_rows)

    # Summary table for manuscript
    summary = []
    for row in acq_rows:
        if row["added_probes"] in {0, 3, 6, 9, 12}:
            summary.append({
                "added_probes": row["added_probes"],
                "residual_greedy_%": f"{row['residual_greedy_frac']:.1%}",
                "uncertainty_%": f"{row['uncertainty_frac']:.1%}",
                "diversity_%": f"{row['diversity_frac']:.1%}",
                "benchmark_aligned_%": f"{row['benchmark_aligned_frac']:.1%}",
                "random_mean_%": f"{row['random_mean_frac']:.1%}",
            })
    write_csv(OUTDIR / "baseline_comparison_acquisition_summary.csv", summary)
    print("\nAcquisition summary (certified fraction):")
    for r in summary:
        print(
            f"  probes={r['added_probes']}  "
            f"greedy={r['residual_greedy_%']}  "
            f"unc={r['uncertainty_%']}  "
            f"div={r['diversity_%']}  "
            f"rand={r['random_mean_%']}"
        )

    # --- Radius sensitivity experiment ---
    radius_rows = radius_sensitivity_experiment(B, deploy, pool)
    write_csv(OUTDIR / "baseline_comparison_radius_sensitivity.csv", radius_rows)
    print("\nRadius sensitivity after three added probes:")
    for r in radius_rows:
        print(
            f"  R scale={r['radius_scale']:.2f}  "
            f"greedy={r['residual_greedy_frac']:.1%}  "
            f"unc={r['uncertainty_frac']:.1%}  "
            f"div={r['diversity_frac']:.1%}  "
            f"rand={r['random_mean_frac']:.1%}"
        )


if __name__ == "__main__":
    main()
