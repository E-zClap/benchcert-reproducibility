"""Nonlinear linearization ablation for response-rank certification.

The manuscript's theorem is exact for linear response representations.  This
script stress-tests the local Jacobian extension on randomly initialized tanh
networks, measuring the deployment Taylor remainder and the benchmark-channel
leakage induced by moving along the local benchmark-null residual direction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def make_network(rng: np.random.Generator, d_in: int, width: int, d_out: int, depth: int):
    dims = [d_in] + [width] * depth + [d_out]
    weights = []
    biases = []
    for a, b in zip(dims[:-1], dims[1:]):
        weights.append(rng.normal(scale=1.0 / np.sqrt(a), size=(b, a)))
        biases.append(rng.normal(scale=0.05, size=b))
    return weights, biases


def forward(weights, biases, z):
    x = z
    for w, b in zip(weights[:-1], biases[:-1]):
        x = np.tanh(w @ x + b)
    return weights[-1] @ x + biases[-1]


def grad_scalar(weights, biases, z, probe):
    activations = [z]
    preacts = []
    x = z
    for w, b in zip(weights[:-1], biases[:-1]):
        a = w @ x + b
        preacts.append(a)
        x = np.tanh(a)
        activations.append(x)
    _ = weights[-1] @ x + biases[-1]
    v = weights[-1].T @ probe
    for w, a in reversed(list(zip(weights[:-1], preacts))):
        v = w.T @ (v * (1.0 - np.tanh(a) ** 2))
    return v


def scalar_response(weights, biases, z, probe):
    return float(probe @ forward(weights, biases, z))


def projector_row_space(rows: np.ndarray) -> np.ndarray:
    # Project in input/tangent space onto the span of benchmark gradients.
    q, _ = np.linalg.qr(rows.T, mode="reduced")
    return q @ q.T


def empirical_hessian_envelope(weights, biases, z, probe, rng, n_dirs=96, eps=1e-3):
    y0 = scalar_response(weights, biases, z, probe)
    vals = []
    for _ in range(n_dirs):
        u = rng.normal(size=z.shape)
        u /= np.linalg.norm(u)
        yp = scalar_response(weights, biases, z + eps * u, probe)
        ym = scalar_response(weights, biases, z - eps * u, probe)
        vals.append(abs(yp - 2.0 * y0 + ym) / (eps**2))
    return float(max(vals))


def run(seed_count=50):
    rng_master = np.random.default_rng(20260425)
    d_in = 12
    d_out = 8
    width = 32
    bench_probes = np.eye(d_out)[:4]
    kstar = np.zeros(d_out)
    kstar[:4] = 0.25
    kstar[4:] = 0.5
    kstar /= np.linalg.norm(kstar)
    depths = [1, 2, 4, 6]
    radii = [0.025, 0.05, 0.10, 0.20]
    rows = []
    for depth in depths:
        for seed in range(seed_count):
            rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
            weights, biases = make_network(rng, d_in, width, d_out, depth)
            z = rng.normal(size=d_in)
            j_b = np.vstack([grad_scalar(weights, biases, z, p) for p in bench_probes])
            j_star = grad_scalar(weights, biases, z, kstar)
            p_b = projector_row_space(j_b)
            residual = (np.eye(d_in) - p_b) @ j_star
            g_j = float(np.linalg.norm(residual))
            if g_j < 1e-10:
                continue
            direction = residual / g_j
            h_star = empirical_hessian_envelope(weights, biases, z, kstar, rng)
            h_b = max(
                empirical_hessian_envelope(weights, biases, z, p, rng, n_dirs=48)
                for p in bench_probes
            )
            for radius in radii:
                dz = radius * direction
                exact_deploy = (
                    scalar_response(weights, biases, z + dz, kstar)
                    - scalar_response(weights, biases, z, kstar)
                )
                linear_deploy = float(j_star @ dz)
                deploy_remainder = abs(exact_deploy - linear_deploy)
                deploy_bound = 0.5 * h_star * radius**2
                bench_leakage = max(
                    abs(
                        scalar_response(weights, biases, z + dz, p)
                        - scalar_response(weights, biases, z, p)
                    )
                    for p in bench_probes
                )
                bench_bound = 0.5 * h_b * radius**2
                rows.append(
                    {
                        "depth": depth,
                        "seed": seed,
                        "radius": radius,
                        "g_jacobian": g_j,
                        "linear_hidden_change": radius * g_j,
                        "exact_hidden_change": abs(exact_deploy),
                        "deployment_remainder": deploy_remainder,
                        "deployment_bound": deploy_bound,
                        "deployment_bound_2x": 2.0 * deploy_bound,
                        "deployment_bound_covers": deploy_remainder <= deploy_bound + 1e-10,
                        "deployment_bound_2x_covers": deploy_remainder <= 2.0 * deploy_bound + 1e-10,
                        "benchmark_leakage": bench_leakage,
                        "benchmark_leakage_bound": bench_bound,
                        "benchmark_leakage_bound_2x": 2.0 * bench_bound,
                        "benchmark_bound_covers": bench_leakage <= bench_bound + 1e-10,
                        "benchmark_bound_2x_covers": bench_leakage <= 2.0 * bench_bound + 1e-10,
                        "relative_deployment_remainder": deploy_remainder
                        / max(abs(linear_deploy), 1e-12),
                        "relative_benchmark_leakage": bench_leakage
                        / max(abs(linear_deploy), 1e-12),
                    }
                )
    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["depth", "radius"], as_index=False)
        .agg(
            mean_g_jacobian=("g_jacobian", "mean"),
            mean_linear_hidden_change=("linear_hidden_change", "mean"),
            mean_exact_hidden_change=("exact_hidden_change", "mean"),
            median_relative_deployment_remainder=("relative_deployment_remainder", "median"),
            q90_relative_deployment_remainder=("relative_deployment_remainder", lambda x: x.quantile(0.9)),
            median_relative_benchmark_leakage=("relative_benchmark_leakage", "median"),
            q90_relative_benchmark_leakage=("relative_benchmark_leakage", lambda x: x.quantile(0.9)),
            deployment_bound_coverage=("deployment_bound_covers", "mean"),
            benchmark_bound_coverage=("benchmark_bound_covers", "mean"),
            deployment_bound_2x_coverage=("deployment_bound_2x_covers", "mean"),
            benchmark_bound_2x_coverage=("benchmark_bound_2x_covers", "mean"),
        )
    )
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "nonlinear_linearization_ablation.csv", index=False)
    summary.to_csv(OUT / "nonlinear_linearization_ablation_summary.csv", index=False)
    return df, summary


if __name__ == "__main__":
    _, summary_df = run()
    print(summary_df.to_string(index=False))
