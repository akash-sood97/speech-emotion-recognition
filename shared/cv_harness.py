"""
Speaker-grouped cross-validation harness for tabular models.

Every classical and embedding-head model goes through ``run_cv``, so
leakage rules stay in one place. Evaluation's held-out exam uses
``fit_tuned`` on the full development set after model choice is frozen.

Paths default to the Modelling section under ``Group-10/``.
Call ``configure_paths`` if another section needs different folders.
"""

from __future__ import annotations

import csv
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import audio_utils as U
import speaker_split as S

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
N_OUTER = 6
EMOTIONS = list(U.EMOTION_ORDER)

_MODELLING = U.submission_root() / "Modelling"
PREDS_DIR = _MODELLING / "outputs" / "preds"
TUNING_LOG = _MODELLING / "outputs" / "tables" / "tuning_log.csv"
FEATURES_PATH = U.features_dir() / "features.parquet"


def configure_paths(
    preds_dir: Path | None = None,
    tuning_log: Path | None = None,
    features_path: Path | None = None,
) -> None:
    """Override output paths (e.g. Evaluation held-out writes)."""
    global PREDS_DIR, TUNING_LOG, FEATURES_PATH
    if preds_dir is not None:
        PREDS_DIR = Path(preds_dir)
    if tuning_log is not None:
        TUNING_LOG = Path(tuning_log)
    if features_path is not None:
        FEATURES_PATH = Path(features_path)


def load_feature_frame(
    path: Path | None = None,
    columns: list[str] | None = None,
):
    """Load development rows of the feature table.

    Parameters
    ----------
    path : Path, optional
        Feature parquet. Default: EDA ``features.parquet``.
    columns : list of str, optional
        Feature columns to keep.

    Returns
    -------
    X, y, groups, fps, cols
        Feature matrix (float64), emotion labels, speaker ids,
        file paths, and column names kept.
    """
    path = FEATURES_PATH if path is None else Path(path)
    feats = pd.read_parquet(path)
    index = U.load_index()
    assert (feats["filepath"].values == index["filepath"].values).all()
    m = S.dev_mask(index)
    cols = columns or [
        c for c in feats.columns if c not in ("filepath", "emotion")
    ]
    X = np.asarray(feats.loc[m, cols].to_numpy(dtype=np.float64))
    y = np.asarray(index.loc[m, "emotion"].to_numpy(), dtype=object)
    groups = np.asarray(index.loc[m, "speaker"].to_numpy(), dtype=np.int64)
    fps = np.asarray(index.loc[m, "filepath"].to_numpy(), dtype=object)
    S.assert_no_test_speakers(fps, path.name)
    return X, y, groups, fps, cols


def outer_splits(split_type: str, X, y, groups):
    """Return the six outer train/validation index pairs."""
    if split_type == "grouped":
        return list(
            S.get_dev_folds(n_splits=N_OUTER).split(X, y, groups=groups)
        )
    if split_type == "random":
        return list(
            S.get_random_folds(n_splits=N_OUTER, seed=SEED).split(X, y)
        )
    raise ValueError(split_type)


def speaker_zscore(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Z-score every column within each speaker (label-free)."""
    out = np.empty_like(X)
    for g in np.unique(groups):
        m = groups == g
        mu = np.nanmean(X[m], axis=0)
        sd = np.nanstd(X[m], axis=0)
        sd[sd == 0] = 1.0
        out[m] = (X[m] - mu) / sd
    return out


def random_oversample(X, y, groups, seed=SEED):
    """Upsample minority classes on the training fold only."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    target = counts.max()
    idx = [np.arange(len(y))]
    for c, n in zip(classes, counts):
        if n < target:
            pool = np.flatnonzero(y == c)
            idx.append(rng.choice(pool, size=target - n, replace=True))
    idx = np.concatenate(idx)
    return X[idx], y[idx], groups[idx]


def save_preds(path: Path, fps, y_true, proba: np.ndarray, classes):
    """Write one fold's predictions with per-class probabilities."""
    proba = pd.DataFrame(proba, columns=list(classes))[EMOTIONS].values
    pred = np.array(EMOTIONS)[proba.argmax(axis=1)]
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        ["filepath", "true_emotion", "pred_emotion", "pred_confidence"]
        + [f"p_{e}" for e in EMOTIONS]
    )
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for fp, t, p, pr in zip(fps, y_true, pred, proba):
            w.writerow(
                [fp, t, p, round(float(pr.max()), 4)]
                + [round(float(v), 5) for v in pr]
            )
    return pred


def log_tuning(row: dict):
    """Append one outer-fold tuning row to the tuning log."""
    TUNING_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not TUNING_LOG.exists()
    fields = [
        "timestamp", "model", "split_type", "fold", "best_params",
        "inner_macro_f1", "outer_macro_f1", "n_train", "n_val",
        "seconds", "note",
    ]
    with open(TUNING_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def run_cv(
    name: str,
    estimator,
    X,
    y,
    groups,
    fps,
    split_type: str,
    param_grid: dict | None = None,
    oversample: bool = False,
    speaker_norm: bool = False,
    refit_estimator=None,
    sample_weight: bool = False,
    write: bool = True,
    verbose: bool = True,
    note: str = "",
) -> list[float]:
    """Run outer CV with optional inner-CV tuning.

    Returns
    -------
    list of float
        Per-fold macro-F1 on the validation speakers.
    """
    from sklearn.utils.class_weight import compute_sample_weight

    if speaker_norm:
        X = speaker_zscore(X, groups)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    scores = []
    for k, (tr, va) in enumerate(outer_splits(split_type, X, y, groups)):
        t0 = time.time()
        Xtr, ytr, gtr = X[tr], y[tr], groups[tr]
        if oversample:
            Xtr, ytr, gtr = random_oversample(
                Xtr, ytr, gtr, seed=SEED + k
            )

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clone(estimator)),
        ])
        fit_kw = {}
        if sample_weight:
            fit_kw["clf__sample_weight"] = compute_sample_weight(
                "balanced", ytr
            )

        best_params, inner = {}, float("nan")
        if param_grid:
            gs = GridSearchCV(
                pipe,
                {f"clf__{k_}": v for k_, v in param_grid.items()},
                scoring="f1_macro",
                cv=S.get_inner_folds(split_type, 3, SEED),
                n_jobs=1,
                refit=False,
            )
            gs.fit(
                Xtr, ytr,
                groups=gtr if split_type == "grouped" else None,
                **fit_kw,
            )
            best_params = {
                k_.replace("clf__", ""): v
                for k_, v in gs.best_params_.items()
            }
            inner = float(gs.best_score_)

        final = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clone(refit_estimator or estimator).set_params(
                **best_params
            )),
        ])
        final.fit(Xtr, ytr, **fit_kw)
        proba = final.predict_proba(X[va])
        classes = final.classes_
        pred = np.array(classes)[proba.argmax(axis=1)]
        f1 = f1_score(
            y[va], pred, average="macro", labels=EMOTIONS, zero_division=0
        )
        scores.append(f1)
        if write:
            save_preds(
                PREDS_DIR / f"{name}_{split_type}_fold{k}.csv",
                fps[va], y[va], proba, classes,
            )
        log_tuning({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": name,
            "split_type": split_type,
            "fold": k,
            "best_params": json.dumps(best_params, default=str),
            "inner_macro_f1": round(inner, 4),
            "outer_macro_f1": round(f1, 4),
            "n_train": len(ytr),
            "n_val": len(va),
            "seconds": round(time.time() - t0, 1),
            "note": note,
        })
        if verbose:
            print(
                f"  {name:<22} {split_type:<7} fold{k}  "
                f"macro-F1 {f1:.4f}  {best_params}",
                flush=True,
            )
    if verbose:
        print(
            f"  {name:<22} {split_type:<7} mean "
            f"{np.mean(scores):.4f} ± {np.std(scores, ddof=1):.4f}",
            flush=True,
        )
    return scores


def fit_tuned(
    estimator,
    X,
    y,
    groups,
    param_grid: dict | None = None,
    refit_estimator=None,
    sample_weight: bool = False,
    inner_splits: int = 3,
):
    """Fit one model on all given rows (full development set).

    Inner speaker-grouped CV picks hyperparameters, then refit on
    everything. Used by the held-out exam after selection is frozen.

    Returns
    -------
    pipe, best_params, inner_macro_f1
    """
    from sklearn.utils.class_weight import compute_sample_weight

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clone(estimator)),
    ])
    fit_kw = {}
    if sample_weight:
        fit_kw["clf__sample_weight"] = compute_sample_weight("balanced", y)
    best_params, inner = {}, float("nan")
    if param_grid:
        gs = GridSearchCV(
            pipe,
            {f"clf__{k}": v for k, v in param_grid.items()},
            scoring="f1_macro",
            cv=S.get_inner_folds("grouped", inner_splits, SEED),
            n_jobs=1,
            refit=False,
        )
        gs.fit(X, y, groups=groups, **fit_kw)
        best_params = {
            k.replace("clf__", ""): v for k, v in gs.best_params_.items()
        }
        inner = float(gs.best_score_)
    final = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clone(refit_estimator or estimator).set_params(
            **best_params
        )),
    ])
    final.fit(X, y, **fit_kw)
    return final, best_params, inner
