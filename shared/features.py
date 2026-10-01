"""
Turn every clip into 512 numbers: the acoustic feature table.

Read by ``02_Feature_engineering`` (which explains each group) and by
the modelling notebooks (which train on the table).

Feature groups (column counts in brackets)
------------------------------------------
energy (5)         loudness of the speech and how it varies over time
pitch (5)          voice pitch (F0) level, spread, range and slope,
                   z-scored per speaker, plus the share of voiced frames
mfcc (480)         40 MFCCs (the timbre, or shape, of the voice) and
                   their first and second time-differences, each
                   summarised by mean, std, min and max
spectral (20)      brightness, roll-off, bandwidth, noisiness and
                   zero-crossing rate, each summarised the same way
voice_quality (2)  jitter and shimmer: frame-to-frame instability of
                   pitch and of loudness

How each group is computed, and why
-----------------------------------
* Energy, pitch and voice quality are measured on the trimmed speech,
  so no silence or padding enters them. They do not depend on the
  window length.
* MFCC and spectral statistics are taken over speech frames only when
  ``masked=True`` (the default): the trimmed signal, centre-cropped if
  longer than the window and never padded. With ``masked=False`` they
  are taken over the fixed window with its zero padding. H12 in
  02_Feature_engineering compares the two.
* Pitch is z-scored within each speaker because pitch varies more
  between speakers than between emotions (H11 in 01_EDA).

Output: ``outputs/features/features.parquet`` (one row per clip, rows
in label-index order) and ``feature_manifest.json`` in the EDA section.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

import audio_utils as U

SR = U.TARGET_RATE
N_MFCC = 40
SEED = 42
PITCH_COLS = ["f0_mean", "f0_std", "f0_range", "f0_slope"]

# Speech frames only: the setting chosen from the window test (H12).
MASKED_DEFAULT = True

# Column-name prefixes that define each feature group.
FEATURE_GROUPS = {
    "mfcc": ("mfcc",),
    "spectral": ("centroid", "rolloff", "bandwidth", "flatness", "zcr"),
    "energy": ("energy",),
    "pitch": ("f0_",),
    "voice_quality": ("jitter", "shimmer"),
}


def feature_group(col: str) -> str:
    """Return the group a feature column belongs to, e.g. ``"mfcc"``."""
    for group, prefixes in FEATURE_GROUPS.items():
        if col.startswith(prefixes):
            return group
    raise KeyError(col)


# ------------------------------------------------------------------- #
# Per-clip feature blocks
# ------------------------------------------------------------------- #

def _agg(name: str, arr: np.ndarray) -> dict[str, float]:
    """Summarise a frame-level series by its mean, std, min and max.

    Non-finite frames are dropped first. An empty series gives NaN.
    """
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"{name}_{s}": np.nan for s in ("mean", "std", "min", "max")}
    return {
        f"{name}_mean": float(np.mean(arr)),
        f"{name}_std": float(np.std(arr)),
        f"{name}_min": float(np.min(arr)),
        f"{name}_max": float(np.max(arr)),
    }


def _mfcc_block(x: np.ndarray) -> dict[str, float]:
    """Return 480 MFCC columns: 40 coefficients x 3 x 4 statistics.

    The three series are the MFCCs themselves, their delta (how fast
    the timbre changes) and delta-delta (how that change accelerates).
    """
    mfcc = librosa.feature.mfcc(y=x, sr=SR, n_mfcc=N_MFCC)
    if mfcc.shape[1] >= 9:
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
    else:
        # A very short speech-only clip has fewer than 9 frames, the
        # default delta window, so use the largest odd window that fits.
        width = max(3, mfcc.shape[1] - (1 - mfcc.shape[1] % 2))
        delta = librosa.feature.delta(mfcc, width=width, mode="nearest")
        delta2 = librosa.feature.delta(mfcc, order=2, width=width,
                                       mode="nearest")
    out: dict[str, float] = {}
    for tag, mat in [("mfcc", mfcc), ("mfcc_delta", delta),
                     ("mfcc_delta2", delta2)]:
        for i in range(mat.shape[0]):
            out.update(_agg(f"{tag}_{i:02d}", mat[i]))
    return out


def _spectral_block(x: np.ndarray) -> dict[str, float]:
    """Return 20 spectral columns: 5 descriptors x 4 statistics.

    centroid   where the energy sits in frequency ("brightness"), Hz
    rolloff    frequency below which 85% of the energy lies, Hz
    bandwidth  spread of the energy around the centroid, Hz
    flatness   tone-like (0) versus noise-like (1)
    zcr        how often the waveform crosses zero, per sample
    """
    out: dict[str, float] = {}
    series = {
        "centroid": lambda: librosa.feature.spectral_centroid(y=x, sr=SR),
        "rolloff": lambda: librosa.feature.spectral_rolloff(y=x, sr=SR),
        "bandwidth": lambda: librosa.feature.spectral_bandwidth(y=x, sr=SR),
        "flatness": lambda: librosa.feature.spectral_flatness(y=x),
        "zcr": lambda: librosa.feature.zero_crossing_rate(x),
    }
    for name, fn in series.items():
        out.update(_agg(name, fn()[0]))
    return out


def _pitch_block(x: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    """Return pitch statistics over voiced frames, and those frames.

    pyin estimates F0 (Hz) per frame and flags which frames are
    voiced; only voiced frames enter the statistics. ``f0_voiced_frac``
    is voiced frames divided by all frames.
    """
    f0, voiced, _ = librosa.pyin(x, fmin=U.PITCH_FMIN, fmax=U.PITCH_FMAX,
                                 sr=SR)
    n_frames = len(f0) if f0 is not None else 0
    v = (f0[voiced].astype(np.float64) if n_frames
         else np.array([], dtype=np.float64))
    if v.size == 0:
        return {"f0_mean": np.nan, "f0_std": np.nan, "f0_range": np.nan,
                "f0_slope": np.nan, "f0_voiced_frac": 0.0}, v
    # Slope of a straight line through the voiced F0 values: a rising
    # or falling intonation over the utterance, in Hz per voiced frame.
    t = np.arange(v.size, dtype=np.float64)
    slope = float(np.polyfit(t, v, 1)[0]) if v.size >= 2 else 0.0
    return {
        "f0_mean": float(np.mean(v)),
        "f0_std": float(np.std(v)),
        "f0_range": float(np.max(v) - np.min(v)),
        "f0_slope": slope,
        "f0_voiced_frac": float(v.size) / max(n_frames, 1),
    }, v


def _jitter_shimmer(x: np.ndarray, f0_frames: np.ndarray
                    ) -> dict[str, float]:
    """Return frame-level jitter and shimmer.

    jitter   mean absolute change in F0 between voiced frames, relative
             to mean F0: how unsteady the pitch is
    shimmer  mean absolute change in frame loudness, relative to mean
             loudness: how unsteady the loudness is

    Both are frame-level approximations, not cycle-by-cycle measures.
    """
    if f0_frames.size < 3:
        jitter = np.nan
    else:
        jitter = float(np.mean(np.abs(np.diff(f0_frames)))
                       / (np.mean(f0_frames) + 1e-8))
    rms = librosa.feature.rms(y=x, frame_length=512, hop_length=256)[0]
    if rms.size < 3:
        shimmer = np.nan
    else:
        shimmer = float(np.mean(np.abs(np.diff(rms)))
                        / (np.mean(rms) + 1e-8))
    return {"jitter": jitter, "shimmer": shimmer}


# ------------------------------------------------------------------- #
# Worker functions. They live in this .py file because worker
# processes can only run functions they can import.
# ------------------------------------------------------------------- #

def _trim_block_one(args):
    """Energy, pitch and voice quality for one clip (trimmed speech)."""
    i, rel, top_db = args
    x = U.trimmed_signal(U.project_root() / rel, top_db=top_db)
    feat = {"energy_rms": float(np.sqrt((x ** 2).mean())) if len(x)
            else 0.0}
    feat.update(_agg("energy_frame_rms", librosa.feature.rms(y=x)[0]))
    pitch, f0 = _pitch_block(x)
    feat.update(pitch)
    feat.update(_jitter_shimmer(x, f0))
    feat["trimmed_sec"] = len(x) / SR
    return i, feat


def _window_block_one(args):
    """MFCC and spectral statistics for one clip, for a given window."""
    i, rel, clip_seconds, top_db, masked = args
    n = int(round(clip_seconds * SR))
    x = U.trimmed_signal(U.project_root() / rel, top_db=top_db)
    if masked:
        # Speech frames only: crop a long clip, never pad a short one.
        x = x if len(x) <= n else U.fix_length(x, n)
    else:
        # Fixed window: crop or zero-pad, exactly as the cache does.
        x = U.fix_length(x, n)
    # Round-trip through float16 so the precision matches the cache.
    x = x.astype(U.CACHE_DTYPE).astype(np.float32)
    feat = {}
    feat.update(_mfcc_block(x))
    feat.update(_spectral_block(x))
    return i, feat


def _run_pool(fn, jobs, n_workers, label):
    """Run ``fn`` over ``jobs`` in parallel; return rows in job order."""
    out = [None] * len(jobs)
    with mp.Pool(n_workers) as pool:
        stream = pool.imap_unordered(fn, jobs, chunksize=8)
        for k, (i, feat) in enumerate(stream, 1):
            out[i] = feat
            if k % 360 == 0:
                print(f"  {label}: {k}/{len(jobs)}", flush=True)
    return pd.DataFrame(out)


# ------------------------------------------------------------------- #
# Public functions
# ------------------------------------------------------------------- #

def trim_block(top_db: float, n_workers: int = 8,
               refresh: bool = False) -> pd.DataFrame:
    """Return the window-independent block (energy, pitch, voice quality).

    pyin makes this the slow part (about 4 minutes on 8 cores), so the
    result is kept in ``cache/`` and reused for every window length.

    Parameters
    ----------
    top_db : float
        Trim threshold in dB.
    refresh : bool
        Recompute even if a cached copy exists.
    """
    path = U.cache_dir() / f"trim_block_trim{int(top_db)}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    index = U.load_index()
    jobs = [(i, rel, top_db) for i, rel in enumerate(index["filepath"])]
    df = _run_pool(_trim_block_one, jobs, n_workers,
                   f"energy/pitch trim {int(top_db)} dB")
    df.to_parquet(path, index=False)
    return df


def extract_all(clip_seconds: float = None, top_db: float = None,
                masked: bool = MASKED_DEFAULT, out_path: Path = None,
                n_workers: int = 8, verbose: bool = True):
    """Build the full feature table for every clip.

    Parameters
    ----------
    clip_seconds : float, optional
        Window for the MFCC and spectral statistics (default 3.5 s).
    top_db : float, optional
        Trim threshold (default 30 dB).
    masked : bool
        True: MFCC/spectral statistics over speech frames only.
        False: over the fixed window including zero padding.
    out_path : Path, optional
        Where to write the table. Default:
        ``outputs/features/features.parquet``, plus
        ``feature_manifest.json`` beside it.

    Returns
    -------
    tuple
        ``(features DataFrame, manifest dict)``.
    """
    clip_seconds = U.CLIP_SECONDS if clip_seconds is None else clip_seconds
    top_db = U.TRIM_TOP_DB if top_db is None else top_db
    default_out = out_path is None
    out_path = (U.features_dir() / "features.parquet" if default_out
                else Path(out_path))
    np.random.seed(SEED)
    index = U.load_index()

    # 1. The two blocks: trimmed-speech measures and window statistics.
    tb = trim_block(top_db, n_workers)
    jobs = [(i, rel, clip_seconds, top_db, masked)
            for i, rel in enumerate(index["filepath"])]
    wb = _run_pool(_window_block_one, jobs, n_workers,
                   f"mfcc/spectral {clip_seconds} s")

    # 2. Assemble one row per clip, in label-index order.
    features = pd.DataFrame({"filepath": index["filepath"],
                             "emotion": index["emotion"]})
    features = pd.concat([features, wb], axis=1)
    features["energy_rms"] = tb["energy_rms"].values
    for c in [c for c in tb.columns if c.startswith("energy_frame_rms")]:
        features[c] = tb[c].values
    features["jitter"] = tb["jitter"].values
    features["shimmer"] = tb["shimmer"].values

    # 3. Pitch z-scored within each speaker. It uses no labels, only
    # each speaker's own clips. Deployment assumption: a baseline of
    # the caller's pitch is available.
    spk = index["speaker"].values
    for col in PITCH_COLS:
        s = tb[col]
        mu = s.groupby(spk).transform("mean")
        sd = s.groupby(spk).transform("std").replace(0, np.nan)
        features[f"{col}_speaker_z"] = ((s - mu) / sd).values
    features["f0_voiced_frac"] = tb["f0_voiced_frac"].values

    # 4. Manifest: exactly how this table was made.
    feature_cols = [c for c in features.columns
                    if c not in ("filepath", "emotion")]
    groups = {g: sum(feature_group(c) == g for c in feature_cols)
              for g in FEATURE_GROUPS}
    n = int(round(clip_seconds * SR))
    speech = tb["trimmed_sec"].values * SR
    manifest = {
        "generated_by": "shared/features.py",
        "seed": SEED,
        "n_clips": len(features),
        "n_feature_columns": len(feature_cols),
        "feature_groups": groups,
        "clip_seconds": clip_seconds,
        "trim_top_db": top_db,
        "masked": masked,
        "mean_padding_frac_of_window": round(float(np.mean(
            np.clip(1 - speech / n, 0, 1))), 4),
        "clips_cropped": int((speech > n).sum()),
        "pitch_normalisation": ("per-speaker z-score on "
                                "f0_mean/std/range/slope"),
        "energy_source": ("RMS of the trimmed signal + frame RMS on "
                          "the trimmed signal"),
        "mfcc_spectral_source": (
            "speech frames only (trimmed, cropped to window, not padded)"
            if masked else "fixed-length window incl. zero padding"),
        "metadata_excluded": [
            "speaker (grouping variable only)",
            "filename fields 2-4 (not decoded, per the project brief)"],
    }

    # 5. Write the table and its manifest.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)
    manifest_path = (out_path.with_name("feature_manifest.json")
                     if default_out else
                     out_path.with_name(out_path.stem + "_manifest.json"))
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    if verbose:
        print(f"wrote {out_path.name}: {len(feature_cols)} feature columns")
    return features, manifest
