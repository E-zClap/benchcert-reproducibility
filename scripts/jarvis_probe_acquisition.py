"""Prospective probe-acquisition experiment on JARVIS DFT data.

Materials-domain analogue of prospective_probe_acquisition_tox21.py.

Protocol
--------
* Dataset   : 187 JARVIS materials with formation_energy_peratom,
              optb88vdw_bandgap AND mbj_bandgap (hybrid-functional band gap).
* Benchmark : formation_energy_peratom  (publicly reported DFT property).
* Deployment: optb88vdw_bandgap > 1.0 eV threshold.
* Initial panel: {formation_energy_peratom}.
* Candidate probes (choose one to add):
    - mbj_bandgap        real JARVIS hybrid-functional gap  (r = 0.98 with bg)
    - fe_control         synthetic probe: fe + N(0, 0.3*sigma_fe)
                         (mimics a benchmark-aligned / formation-energy-like
                          measurement that a benchmark-aligned policy would pick)
* Policies:
    - response_rank      picks the probe with highest |r(probe, bg_residual)|^2
    - benchmark_aligned  picks the probe most correlated with formation energy
    - oracle             picks whichever probe gives higher held-out certification
                         (post-hoc upper bound, not a fair policy)
    - random             uniform random choice (50/50 between two probes)

Calibration/test split: 70/30, repeated over N_SEEDS independent seeds.
Probe choice is made from calibration compounds ONLY, frozen before
held-out test materials are evaluated.

Certification mechanism: 2-D quantile fiber (benchmark x probe) with
minimum fiber support of 3 calibration materials. A test material is
certified when its calibration fiber is unanimously one label.

Outputs
-------
outputs/jarvis_probe_acquisition_trace.csv
outputs/jarvis_probe_acquisition_summary.csv
"""

from __future__ import annotations

import csv
import io
import json
import math
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(exist_ok=True)
CACHEDIR = ROOT / ".cache" / "downloads"
CACHEDIR.mkdir(parents=True, exist_ok=True)

RAW_BASE = "https://raw.githubusercontent.com/usnistgov/jarvis_leaderboard/main"
BENCH_BASE = "jarvis_leaderboard/benchmarks/AI/SinglePropertyPrediction"

DEPLOYMENT_THRESHOLD = 1.0   # eV  (optb88vdw_bandgap > 1.0 → "viable")
CAL_FRACTION = 0.70
MIN_FIBER_SIZE = 3
N_BINS = 8          # quantile bins per dimension in 2-D fiber
N_SEEDS = 200
NOISE_SCALE = 0.30  # fraction of fe std for synthetic control probe


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _fetch(url: str, timeout: int = 120) -> bytes:
    import hashlib
    key = hashlib.sha256(url.encode()).hexdigest()
    cache_path = CACHEDIR / f"{key}.bin"
    if cache_path.exists():
        return cache_path.read_bytes()
    req = Request(url, headers={"User-Agent": "Codex"})
    with urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    cache_path.write_bytes(data)
    return data


def _load_reference(prop_name: str) -> dict[str, float]:
    url = f"{RAW_BASE}/{BENCH_BASE}/dft_3d_{prop_name}.json.zip"
    raw = _fetch(url)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        obj = json.loads(zf.read(zf.namelist()[0]).decode())
    raw_map = obj["test"] if ("test" in obj and isinstance(obj["test"], dict)) else obj
    out: dict[str, float] = {}
    for jid, val in raw_map.items():
        if isinstance(val, (int, float)):
            v = float(val)
            if math.isfinite(v):
                out[jid] = v
        elif isinstance(val, dict) and prop_name in val:
            v = float(val[prop_name])
            if math.isfinite(v):
                out[jid] = v
    return out


# ---------------------------------------------------------------------------
# Linear geometry helpers
# ---------------------------------------------------------------------------

def _row_span_projector(rows: np.ndarray) -> np.ndarray:
    if rows.ndim == 1:
        rows = rows[np.newaxis]
    q, _ = np.linalg.qr(rows.T)
    rank = np.linalg.matrix_rank(rows, tol=1e-9)
    return q[:, :rank] @ q[:, :rank].T


def _span_residual(bench_cal: np.ndarray, deploy_cal: np.ndarray) -> float:
    P = _row_span_projector(bench_cal[np.newaxis] if bench_cal.ndim == 1 else bench_cal)
    return float(np.linalg.norm(deploy_cal - P @ deploy_cal))


def _response_rank_score(
    bench_cal: np.ndarray, probe_cal: np.ndarray, deploy_cal: np.ndarray
) -> float:
    """Squared cosine of probe with deployment residual after benchmark projection."""
    P = _row_span_projector(bench_cal[np.newaxis])
    r = deploy_cal - P @ deploy_cal
    r_norm = float(np.linalg.norm(r))
    if r_norm < 1e-12:
        return 0.0
    p_unit = probe_cal / (np.linalg.norm(probe_cal) + 1e-12)
    return float(np.dot(p_unit, r / r_norm) ** 2)


def _benchmark_aligned_score(bench_cal: np.ndarray, probe_cal: np.ndarray) -> float:
    """Squared cosine of probe with benchmark (formation energy) direction."""
    b_unit = bench_cal / (np.linalg.norm(bench_cal) + 1e-12)
    p_unit = probe_cal / (np.linalg.norm(probe_cal) + 1e-12)
    return float(np.dot(b_unit, p_unit) ** 2)


# ---------------------------------------------------------------------------
# Fiber certification
# ---------------------------------------------------------------------------

def _certify_2d(
    bench_cal: np.ndarray,
    probe_cal: np.ndarray,
    labels_cal: np.ndarray,
    bench_test: np.ndarray,
    probe_test: np.ndarray,
    labels_test: np.ndarray,
    n_bins: int = N_BINS,
    min_size: int = MIN_FIBER_SIZE,
) -> tuple[float, float]:
    """Return (certified_fraction, false_decision_rate) on test set."""
    q_b = np.unique(np.quantile(bench_cal, np.linspace(0, 1, n_bins + 1)))
    q_p = np.unique(np.quantile(probe_cal, np.linspace(0, 1, n_bins + 1)))

    def idx(v: np.ndarray, cuts: np.ndarray) -> np.ndarray:
        return np.searchsorted(cuts[1:-1], v, side="right")

    fiber_labs: dict[tuple[int, int], list[int]] = defaultdict(list)
    for b, p, lab in zip(idx(bench_cal, q_b), idx(probe_cal, q_p), labels_cal):
        fiber_labs[(int(b), int(p))].append(int(lab))

    fiber_cert: dict[tuple[int, int], int | None] = {}
    for key, labs in fiber_labs.items():
        if len(labs) < min_size:
            fiber_cert[key] = None
        elif all(l == labs[0] for l in labs):
            fiber_cert[key] = labs[0]
        else:
            fiber_cert[key] = None

    n_test = len(labels_test)
    certified = false_dec = 0
    for b, p, true_lab in zip(
        idx(bench_test, q_b), idx(probe_test, q_p), labels_test
    ):
        cert = fiber_cert.get((int(b), int(p)))
        if cert is not None:
            certified += 1
            if cert != int(true_lab):
                false_dec += 1
    return certified / n_test, false_dec / n_test


def _certify_1d(
    bench_cal: np.ndarray,
    labels_cal: np.ndarray,
    bench_test: np.ndarray,
    labels_test: np.ndarray,
    n_bins: int = N_BINS * N_BINS,
    min_size: int = MIN_FIBER_SIZE,
) -> tuple[float, float]:
    """1-D formation-energy-only baseline certification."""
    cuts = np.unique(np.quantile(bench_cal, np.linspace(0, 1, n_bins + 1)))

    def idx(v: np.ndarray) -> np.ndarray:
        return np.searchsorted(cuts[1:-1], v, side="right")

    fiber_labs: dict[int, list[int]] = defaultdict(list)
    for b, lab in zip(idx(bench_cal), labels_cal):
        fiber_labs[int(b)].append(int(lab))

    fiber_cert: dict[int, int | None] = {}
    for key, labs in fiber_labs.items():
        if len(labs) < min_size:
            fiber_cert[key] = None
        elif all(l == labs[0] for l in labs):
            fiber_cert[key] = labs[0]
        else:
            fiber_cert[key] = None

    n_test = len(labels_test)
    certified = false_dec = 0
    for b, true_lab in zip(idx(bench_test), labels_test):
        cert = fiber_cert.get(int(b))
        if cert is not None:
            certified += 1
            if cert != int(true_lab):
                false_dec += 1
    return certified / n_test, false_dec / n_test


# ---------------------------------------------------------------------------
# Single seed
# ---------------------------------------------------------------------------

def _run_seed(
    seed: int,
    fe: np.ndarray,
    bg: np.ndarray,
    mbj: np.ndarray,
    labels: np.ndarray,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    n = len(labels)
    perm = rng.permutation(n)
    n_cal = int(round(CAL_FRACTION * n))
    cal = perm[:n_cal]
    tst = perm[n_cal:]

    # Normalize on calibration set
    def _norm_cal(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu, sigma = v[cal].mean(), v[cal].std() + 1e-12
        return (v[cal] - mu) / sigma, (v[tst] - mu) / sigma

    fe_cal, fe_tst = _norm_cal(fe)
    bg_cal, bg_tst = _norm_cal(bg)
    mbj_cal, mbj_tst = _norm_cal(mbj)

    # Synthetic formation-energy-like control probe
    ctrl_raw = fe + rng.normal(0, NOISE_SCALE * fe.std(), size=n)
    ctrl_cal, ctrl_tst = _norm_cal(ctrl_raw)

    lab_cal = labels[cal]
    lab_tst = labels[tst]

    # --- Baseline: formation energy only (1-D) ---
    cert0, fd0 = _certify_1d(fe_cal, lab_cal, fe_tst, lab_tst)

    rows: list[dict] = []

    probes = {
        "mbj_bandgap": (mbj_cal, mbj_tst),
        "fe_control":  (ctrl_cal, ctrl_tst),
    }

    # --- Span-residual geometry (calibration only, for policy scores) ---
    rr_scores = {
        name: _response_rank_score(fe_cal, vc, bg_cal)
        for name, (vc, _) in probes.items()
    }
    ba_scores = {
        name: _benchmark_aligned_score(fe_cal, vc)
        for name, (vc, _) in probes.items()
    }

    def _cert_with_probe(name: str) -> tuple[float, float]:
        vc_cal, vc_tst = probes[name]
        return _certify_2d(fe_cal, vc_cal, lab_cal, fe_tst, vc_tst, lab_tst)

    oracle_pick = max(probes.keys(), key=lambda n: _cert_with_probe(n)[0])

    for policy in ("response_rank", "benchmark_aligned", "oracle", "random"):
        if policy == "response_rank":
            pick = max(rr_scores, key=rr_scores.__getitem__)
        elif policy == "benchmark_aligned":
            pick = max(ba_scores, key=ba_scores.__getitem__)
        elif policy == "oracle":
            pick = oracle_pick
        elif policy == "random":
            pick = rng.choice(list(probes.keys()))

        cert1, fd1 = _cert_with_probe(pick)
        rows.append({
            "seed": seed,
            "policy": policy,
            "probe_chosen": pick,
            "rr_mbj": rr_scores["mbj_bandgap"],
            "rr_ctrl": rr_scores["fe_control"],
            "ba_mbj": ba_scores["mbj_bandgap"],
            "ba_ctrl": ba_scores["fe_control"],
            "cert_0probe": cert0,
            "cert_1probe": cert1,
            "false_dec_0probe": fd0,
            "false_dec_1probe": fd1,
            "cert_gain": cert1 - cert0,
        })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading JARVIS reference data...", flush=True)
    fe_ref = _load_reference("formation_energy_peratom")
    bg_ref = _load_reference("optb88vdw_bandgap")
    mbj_ref = _load_reference("mbj_bandgap")

    # Intersect to materials with all three properties
    jids = sorted(set(fe_ref) & set(bg_ref) & set(mbj_ref))
    print(f"  Materials with fe + bg + mbj: {len(jids)}", flush=True)
    if len(jids) < 50:
        sys.exit("Too few materials for a meaningful experiment.")

    fe_arr = np.array([fe_ref[j] for j in jids])
    bg_arr = np.array([bg_ref[j] for j in jids])
    mbj_arr = np.array([mbj_ref[j] for j in jids])
    labels = (bg_arr > DEPLOYMENT_THRESHOLD).astype(int)

    pos_rate = labels.mean()
    print(f"  Deployment threshold {DEPLOYMENT_THRESHOLD} eV: "
          f"{pos_rate:.1%} viable, {1-pos_rate:.1%} nonviable", flush=True)

    # Quick geometry check (span residual = sqrt(N)*sqrt(1-r^2) for z-scored vectors)
    r_fe_bg = float(np.corrcoef(fe_arr, bg_arr)[0, 1])
    r_mbj_bg = float(np.corrcoef(mbj_arr, bg_arr)[0, 1])
    n_mat = len(fe_arr)
    g0 = float(np.sqrt(n_mat) * np.sqrt(max(0.0, 1.0 - r_fe_bg ** 2)))
    # g after adding mbj: use QR projector on z-scored vectors
    fe_n = (fe_arr - fe_arr.mean()) / fe_arr.std()
    bg_n = (bg_arr - bg_arr.mean()) / bg_arr.std()
    mbj_n = (mbj_arr - mbj_arr.mean()) / mbj_arr.std()
    K2 = np.vstack([fe_n, mbj_n])
    Q, _ = np.linalg.qr(K2.T)
    P2 = Q[:, :2] @ Q[:, :2].T
    g1 = float(np.linalg.norm(bg_n - P2 @ bg_n))
    print(f"  Geometry: r(fe,bg)={r_fe_bg:.3f}, r(mbj,bg)={r_mbj_bg:.3f}")
    print(f"  Span residual: g(fe only)={g0:.3f}, g(fe+mbj)={g1:.3f}, "
          f"reduction={100*(g0-g1)/g0:.1f}%", flush=True)

    print(f"\nRunning {N_SEEDS} calibration/test splits...", flush=True)
    all_rows: list[dict] = []
    for s in range(N_SEEDS):
        all_rows.extend(_run_seed(s, fe_arr, bg_arr, mbj_arr, labels))
        if (s + 1) % 50 == 0:
            print(f"  seed {s+1}/{N_SEEDS}", flush=True)

    # Save trace
    trace_path = OUTDIR / "jarvis_probe_acquisition_trace.csv"
    with open(trace_path, "w", newline="") as f:
        fieldnames = list(all_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nTrace → {trace_path}")

    # Compute summary per policy
    from collections import Counter
    policy_rows: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        policy_rows[r["policy"]].append(r)

    summary: list[dict] = []
    for policy, rows in policy_rows.items():
        n = len(rows)
        cert0 = np.mean([r["cert_0probe"] for r in rows])
        cert1 = np.mean([r["cert_1probe"] for r in rows])
        fd1   = np.mean([r["false_dec_1probe"] for r in rows])
        gain  = np.mean([r["cert_gain"] for r in rows])
        n_mbj = sum(1 for r in rows if r["probe_chosen"] == "mbj_bandgap")
        summary.append({
            "policy": policy,
            "n_seeds": n,
            "cert_0probe_mean": round(cert0, 4),
            "cert_1probe_mean": round(cert1, 4),
            "false_dec_1probe_mean": round(fd1, 5),
            "cert_gain_mean": round(gain, 4),
            "pct_chose_mbj": round(100 * n_mbj / n, 1),
            "pct_chose_fe_control": round(100 * (n - n_mbj) / n, 1),
        })

    summary.sort(key=lambda r: -r["cert_1probe_mean"])

    summary_path = OUTDIR / "jarvis_probe_acquisition_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"Summary → {summary_path}\n")

    # Print table
    print(f"{'Policy':<20} {'Cert_0':>7} {'Cert_1':>7} {'FalseD':>7} "
          f"{'Gain':>7} {'%mbj':>6} {'%ctrl':>6}")
    print("-" * 70)
    for r in summary:
        print(f"{r['policy']:<20} {r['cert_0probe_mean']:>7.3f} "
              f"{r['cert_1probe_mean']:>7.3f} "
              f"{r['false_dec_1probe_mean']:>7.4f} "
              f"{r['cert_gain_mean']:>7.3f} "
              f"{r['pct_chose_mbj']:>6.1f} "
              f"{r['pct_chose_fe_control']:>6.1f}")

    # Also print span-residual reduction by probe for context
    print("\nSpan-residual geometry (full 187-material dataset):")
    print(f"  r(fe, bg)  = {r_fe_bg:.4f}    g before any probe = {g0:.4f}")
    print(f"  r(mbj, bg) = {r_mbj_bg:.4f}   g after mbj probe  = {g1:.4f}  "
          f"(reduction {100*(g0-g1)/g0:.1f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()
