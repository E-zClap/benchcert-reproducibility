"""Mainstream vision robustness audit for the NMI article.

This script uses the public scikit-learn handwritten-digits benchmark as a
small, fully reproducible vision case. The benchmark response is clean-image
classification accuracy. The deployment response is corruption-suite robustness:
a candidate is viable only if the model classifies the clean image and all
corrupted variants correctly. Finite benchmark fibers are defined by clean
prediction, clean correctness and binned clean confidence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, shift
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def corruptions(x: np.ndarray) -> list[np.ndarray]:
    """Return deterministic deployment corruptions for flattened 8x8 images."""
    imgs = x.reshape(-1, 8, 8)
    rng = np.random.default_rng(20260425)
    noise = np.clip(imgs + rng.normal(0, 0.23, size=imgs.shape), 0, 1)
    blur = np.stack([gaussian_filter(im, sigma=0.85) for im in imgs])
    occluded = imgs.copy()
    occluded[:, 2:6, 2:4] = 0
    shifted = np.stack([shift(im, shift=(0.7, -0.7), order=1, mode="nearest") for im in imgs])
    return [noise.reshape(len(x), -1), blur.reshape(len(x), -1), occluded.reshape(len(x), -1), shifted.reshape(len(x), -1)]


def confidence(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x).max(axis=1)
    if hasattr(model, "decision_function"):
        margin = np.asarray(model.decision_function(x))
        if margin.ndim == 1:
            return np.abs(margin)
        top = np.partition(margin, -2, axis=1)[:, -2:]
        return top[:, 1] - top[:, 0]
    return np.ones(len(x))


def make_models(seed: int):
    return {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs"),
        ),
        "linear_svm": make_pipeline(StandardScaler(), LinearSVC(C=0.4, dual="auto", max_iter=5000)),
        "rbf_svm": make_pipeline(StandardScaler(), SVC(C=3.0, gamma="scale", probability=True)),
        "random_forest": RandomForestClassifier(n_estimators=250, max_depth=None, random_state=seed),
        "extra_trees": ExtraTreesClassifier(n_estimators=250, random_state=seed),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(80,), alpha=1e-3, max_iter=600, random_state=seed),
        ),
        "gaussian_nb": GaussianNB(),
    }


def finite_fiber_metrics(pred: np.ndarray, clean_ok: np.ndarray, conf: np.ndarray, robust_ok: np.ndarray):
    bins = pd.qcut(conf, q=10, labels=False, duplicates="drop")
    table = pd.DataFrame(
        {
            "pred": pred,
            "clean_ok": clean_ok.astype(int),
            "conf_bin": np.asarray(bins),
            "robust_ok": robust_ok.astype(int),
        }
    )
    grouped = table.groupby(["pred", "clean_ok", "conf_bin"], dropna=False)["robust_ok"]
    stats = grouped.agg(["count", "min", "max"]).reset_index()
    cert_keys = stats[stats["min"] == stats["max"]][["pred", "clean_ok", "conf_bin"]]
    cert = table.merge(cert_keys.assign(certified=1), how="left", on=["pred", "clean_ok", "conf_bin"])[
        "certified"
    ].fillna(0)
    ambiguous = 1 - cert.to_numpy(dtype=float)
    return {
        "certifiable_fraction": float(cert.mean()),
        "ambiguous_fraction": float(ambiguous.mean()),
        "n_fibers": int(len(stats)),
        "ambiguous_fibers": int((stats["min"] != stats["max"]).sum()),
        "median_fiber_size": float(stats["count"].median()),
    }


def run() -> None:
    digits = load_digits()
    x = digits.data.astype(float) / 16.0
    y = digits.target.astype(int)
    rows = []
    choices = []
    splitter = StratifiedShuffleSplit(n_splits=25, test_size=0.40, random_state=17)

    for split, (train_idx, test_idx) in enumerate(splitter.split(x, y)):
        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]
        corrupted = corruptions(x_test)
        split_rows = []
        for model_name, model in make_models(split).items():
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            clean_ok = pred == y_test
            robust_ok = clean_ok.copy()
            for xc in corrupted:
                robust_ok &= model.predict(xc) == y_test
            conf = confidence(model, x_test)
            metrics = finite_fiber_metrics(pred, clean_ok, conf, robust_ok)
            row = {
                "split": split,
                "model": model_name,
                "clean_accuracy": accuracy_score(y_test, pred),
                "robust_suite_accuracy": float(robust_ok.mean()),
                **metrics,
            }
            rows.append(row)
            split_rows.append(row)
        best_clean = max(split_rows, key=lambda r: r["clean_accuracy"])
        best_cert = max(split_rows, key=lambda r: (r["certifiable_fraction"], r["robust_suite_accuracy"]))
        best_robust = max(split_rows, key=lambda r: r["robust_suite_accuracy"])
        choices.append(
            {
                "split": split,
                "best_clean_model": best_clean["model"],
                "best_clean_accuracy": best_clean["clean_accuracy"],
                "best_clean_certifiable_fraction": best_clean["certifiable_fraction"],
                "best_clean_robust_suite_accuracy": best_clean["robust_suite_accuracy"],
                "best_cert_model": best_cert["model"],
                "best_certifiable_fraction": best_cert["certifiable_fraction"],
                "best_cert_robust_suite_accuracy": best_cert["robust_suite_accuracy"],
                "best_robust_model": best_robust["model"],
                "best_robust_suite_accuracy": best_robust["robust_suite_accuracy"],
                "clean_choice_changed_for_certification": best_clean["model"] != best_cert["model"],
                "clean_choice_changed_for_robustness": best_clean["model"] != best_robust["model"],
            }
        )

    df = pd.DataFrame(rows)
    ch = pd.DataFrame(choices)
    summary = (
        df.groupby("model")
        .agg(
            n_splits=("split", "count"),
            mean_clean_accuracy=("clean_accuracy", "mean"),
            mean_robust_suite_accuracy=("robust_suite_accuracy", "mean"),
            mean_certifiable_fraction=("certifiable_fraction", "mean"),
            mean_ambiguous_fraction=("ambiguous_fraction", "mean"),
            median_fiber_size=("median_fiber_size", "median"),
        )
        .reset_index()
        .sort_values("mean_clean_accuracy", ascending=False)
    )
    overall = pd.DataFrame(
        [
            {
                "n_models": df["model"].nunique(),
                "n_splits": ch["split"].nunique(),
                "median_clean_accuracy": df["clean_accuracy"].median(),
                "median_robust_suite_accuracy": df["robust_suite_accuracy"].median(),
                "median_certifiable_fraction": df["certifiable_fraction"].median(),
                "median_ambiguous_fraction": df["ambiguous_fraction"].median(),
                "clean_choice_changed_for_certification_fraction": ch[
                    "clean_choice_changed_for_certification"
                ].mean(),
                "clean_choice_changed_for_robustness_fraction": ch[
                    "clean_choice_changed_for_robustness"
                ].mean(),
            }
        ]
    )
    df.to_csv(OUT / "vision_robustness_response_audit.csv", index=False)
    summary.to_csv(OUT / "vision_robustness_response_summary_by_model.csv", index=False)
    ch.to_csv(OUT / "vision_robustness_model_choice.csv", index=False)
    overall.to_csv(OUT / "vision_robustness_response_summary.csv", index=False)


if __name__ == "__main__":
    run()
