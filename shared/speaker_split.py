"""
The one speaker split every section uses, and the folds built on it.

Every speaker contributes the same mix of emotions (H6 in 01_EDA). A
random split would therefore put clips of the same voice on both
sides, and a model could score well by recognising who is speaking.
So the audio dataset is split by speaker:

* 6 held-out speakers (360 clips) form the final test set. They are
  opened once, at the very end, and never used for exploring or tuning.
* 18 development ("dev") speakers (1,080 clips) are used for
  everything else, including all of EDA and feature engineering.

Inside the dev set, cross-validation holds out whole speakers too
(6 folds of 3 speakers). A random, speaker-blind version of the folds
is kept only to show what ignoring the speaker would cost.

The split and the outer fold membership are stored in
``frozen_split.json``. ``load_frozen_split()`` re-derives the
held-out speakers from the seed on every call and validates the
stored fold maps, so neither the split nor the folds can change
silently with a different scikit-learn version.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

SEED = 42
N_TEST_SPEAKERS = 6
N_OUTER_FOLDS = 6
N_INNER_FOLDS = 3
ALL_SPEAKERS = list(range(1, 25))  # Person01 ... Person24

FROZEN_SPLIT_JSON = Path(__file__).resolve().parent / "frozen_split.json"


def freeze_split(speaker_ids=ALL_SPEAKERS, seed: int = SEED,
                 n_test: int = N_TEST_SPEAKERS):
    """Draw the held-out speakers with a fixed seed.

    Returns
    -------
    tuple
        ``(test_speakers, dev_speakers)``, both sorted lists of ints.
    """
    rng = random.Random(seed)
    test = sorted(rng.sample(list(speaker_ids), n_test))
    dev = sorted(set(speaker_ids) - set(test))
    return test, dev


def _validate_fold_maps(split: dict) -> None:
    """Stop if stored outer fold maps are incomplete or inconsistent."""
    if "dev_folds" not in split or "random_folds" not in split:
        raise AssertionError(
            "frozen_split.json is missing dev_folds / random_folds. "
            "Do not edit the fold maps: every reported number depends "
            "on them.")
    dev_folds = split["dev_folds"]
    random_folds = split["random_folds"]
    if len(dev_folds) != N_OUTER_FOLDS:
        raise AssertionError(
            f"dev_folds must have {N_OUTER_FOLDS} folds, "
            f"got {len(dev_folds)}")
    flat_sp = sorted(s for fold in dev_folds for s in fold)
    if flat_sp != sorted(split["dev_speakers"]):
        raise AssertionError(
            "dev_folds speakers are not a partition of "
            "dev_speakers")
    for fold in dev_folds:
        if len(fold) != 3 or sorted(fold) != list(fold):
            raise AssertionError(
                f"each dev fold must be 3 sorted speakers; got {fold}")
    if len(random_folds) != N_OUTER_FOLDS:
        raise AssertionError(
            f"random_folds must have {N_OUTER_FOLDS} folds, "
            f"got {len(random_folds)}")
    flat_ix = sorted(i for fold in random_folds for i in fold)
    n_dev = int(split["n_dev_clips"])
    if flat_ix != list(range(n_dev)):
        raise AssertionError(
            "random_folds indices are not a partition of "
            f"0..{n_dev - 1}")
    for fold in random_folds:
        if len(fold) != n_dev // N_OUTER_FOLDS:
            raise AssertionError(
                f"each random fold must cover "
                f"{n_dev // N_OUTER_FOLDS} clips; got {len(fold)}")


def load_frozen_split() -> dict:
    """Return the frozen split, after checking it against the seed.

    Returns
    -------
    dict
        Includes ``test_speakers``, ``dev_speakers``, ``dev_folds``
        (validation speakers per outer grouped fold) and
        ``random_folds`` (validation row indices into the 1,080-row
        development clip order).
    """
    with open(FROZEN_SPLIT_JSON) as f:
        split = json.load(f)
    test, dev = freeze_split(seed=split["seed"])
    if split["test_speakers"] != test or split["dev_speakers"] != dev:
        raise AssertionError(
            "frozen_split.json no longer matches the seed. Do not edit "
            "the split: every reported number depends on it.")
    _validate_fold_maps(split)
    return split


def speaker_of(filepath) -> int:
    """Return the speaker from a path: ``data/Person07/x.wav`` -> 7."""
    for part in Path(str(filepath)).parts:
        if part.startswith("Person"):
            return int(part.replace("Person", ""))
    raise ValueError(f"no PersonNN folder in {filepath!r}")


def dev_mask(index) -> np.ndarray:
    """Return a boolean mask selecting the dev speakers' rows.

    Parameters
    ----------
    index : pandas.DataFrame
        Any per-clip table with a ``speaker`` column.
    """
    dev = set(load_frozen_split()["dev_speakers"])
    return index["speaker"].astype(int).isin(dev).values


def assert_no_test_speakers(filepaths, context: str = "") -> None:
    """Stop the run if any held-out speaker's clip is present.

    Every analysis and training step calls this on the clips it is
    about to use. The held-out speakers are opened only by the final
    test.
    """
    test = set(load_frozen_split()["test_speakers"])
    leaked = sorted({speaker_of(fp) for fp in filepaths} & test)
    if leaked:
        raise AssertionError(
            f"Held-out speaker(s) {leaked} found in {context or 'data'}.")


# ---------------------------------------------------------------------------
# Fold splitters that do not depend on the installed scikit-learn version
# ---------------------------------------------------------------------------

def _group_kfold_test_indices(groups, n_splits: int):
    """Yield test-index arrays; matches scikit-learn 1.9.0 GroupKFold.

    The algorithm is the shuffle=False branch of
    ``sklearn.model_selection.GroupKFold._iter_test_indices`` in
    version 1.9.0 (stable argsort by group size, then lightest-fold
    assignment). Implemented in plain numpy so a different
    scikit-learn install cannot change the folds.
    """
    groups = np.asarray(groups)
    unique_groups, group_idx = np.unique(groups, return_inverse=True)
    n_groups = len(unique_groups)
    if n_splits > n_groups:
        raise ValueError(
            f"Cannot have n_splits={n_splits} greater than "
            f"n_groups={n_groups}")
    n_samples_per_group = np.bincount(group_idx)
    order = np.argsort(n_samples_per_group, kind="stable")[::-1]
    weights = n_samples_per_group[order]
    n_samples_per_fold = np.zeros(n_splits)
    group_to_fold = np.zeros(n_groups)
    for group_index, weight in enumerate(weights):
        lightest = int(np.argmin(n_samples_per_fold))
        n_samples_per_fold[lightest] += weight
        group_to_fold[order[group_index]] = lightest
    fold_of = group_to_fold[group_idx]
    for f in range(n_splits):
        yield np.where(fold_of == f)[0]


def _stratified_kfold_test_folds(y, n_splits: int, seed: int,
                                 shuffle: bool):
    """Return per-sample fold ids; matches sklearn 1.9.0 StratifiedKFold."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    _, y_idx, y_inv = np.unique(y, return_index=True, return_inverse=True)
    _, class_perm = np.unique(y_idx, return_inverse=True)
    y_encoded = class_perm[y_inv]
    n_classes = len(y_idx)
    y_counts = np.bincount(y_encoded)
    if np.all(n_splits > y_counts):
        raise ValueError(
            f"n_splits={n_splits} cannot be greater than the number "
            "of members in each class.")
    y_order = np.sort(y_encoded)
    allocation = np.asarray([
        np.bincount(y_order[i::n_splits], minlength=n_classes)
        for i in range(n_splits)
    ])
    test_folds = np.empty(len(y), dtype="i")
    for k in range(n_classes):
        folds_for_class = np.arange(n_splits).repeat(allocation[:, k])
        if shuffle:
            rng.shuffle(folds_for_class)
        test_folds[y_encoded == k] = folds_for_class
    return test_folds


class _FrozenSpeakerFolds:
    """Outer grouped CV from ``dev_folds`` in frozen_split.json."""

    def __init__(self, n_splits: int = N_OUTER_FOLDS):
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("groups is required for grouped folds")
        groups = np.asarray(groups)
        folds = load_frozen_split()["dev_folds"]
        if len(folds) != self.n_splits:
            raise AssertionError("dev_folds length mismatch")
        indices = np.arange(len(groups))
        for val_speakers in folds:
            val = set(int(s) for s in val_speakers)
            test_mask = np.array([int(g) in val for g in groups])
            test_idx = indices[test_mask]
            train_idx = indices[~test_mask]
            if len(test_idx) == 0:
                raise AssertionError(
                    f"no clips for validation speakers {val_speakers}")
            yield train_idx, test_idx


class _FrozenRandomFolds:
    """Outer random CV from ``random_folds`` in frozen_split.json."""

    def __init__(self, n_splits: int = N_OUTER_FOLDS, seed: int = SEED):
        self.n_splits = n_splits
        self.seed = seed

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n = len(np.asarray(y) if y is not None else X)
        folds = load_frozen_split()["random_folds"]
        n_dev = int(load_frozen_split()["n_dev_clips"])
        if n != n_dev:
            raise AssertionError(
                f"frozen random folds cover {n_dev} development "
                f"clips; got n_samples={n}")
        if len(folds) != self.n_splits:
            raise AssertionError("random_folds length mismatch")
        indices = np.arange(n)
        for test_list in folds:
            test_idx = np.asarray(test_list, dtype=int)
            mask = np.zeros(n, dtype=bool)
            mask[test_idx] = True
            yield indices[~mask], test_idx


class _NumpyGroupKFold:
    """Inner grouped CV; plain-numpy port of sklearn 1.9.0 GroupKFold."""

    def __init__(self, n_splits: int = N_INNER_FOLDS):
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("groups is required for grouped folds")
        groups = np.asarray(groups)
        n = len(groups)
        indices = np.arange(n)
        for test_idx in _group_kfold_test_indices(groups, self.n_splits):
            mask = np.zeros(n, dtype=bool)
            mask[test_idx] = True
            yield indices[~mask], test_idx


class _NumpyStratifiedKFold:
    """Inner helper; numpy port of sklearn 1.9.0 StratifiedKFold."""

    def __init__(self, n_splits: int = N_INNER_FOLDS, shuffle: bool = True,
                 random_state: int = SEED):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if y is None:
            raise ValueError("y is required for stratified folds")
        y = np.asarray(y)
        n = len(y)
        indices = np.arange(n)
        test_folds = _stratified_kfold_test_folds(
            y, self.n_splits, self.random_state, self.shuffle)
        for f in range(self.n_splits):
            test_idx = indices[test_folds == f]
            yield indices[test_folds != f], test_idx


def get_dev_folds(n_splits: int = N_OUTER_FOLDS):
    """Return the speaker-grouped splitter used for all dev results.

    Call ``.split(X, y, groups=speaker_ids)``. Each fold holds out 3
    whole speakers (180 clips) and trains on the other 15 (900 clips).
    Membership comes from ``frozen_split.json`` (``dev_folds``), not
    from the installed scikit-learn version.
    """
    if n_splits != N_OUTER_FOLDS:
        raise ValueError(
            f"only the frozen {N_OUTER_FOLDS}-fold outer split is "
            f"supported; got n_splits={n_splits}")
    return _FrozenSpeakerFolds(n_splits=n_splits)


def get_random_folds(n_splits: int = N_OUTER_FOLDS, seed: int = SEED):
    """Return the speaker-blind splitter, used only for comparison.

    Stratified by emotion and shuffled, ignoring the speaker. Results
    from it are never reported as the real performance. Membership
    comes from ``frozen_split.json`` (``random_folds``).
    """
    if n_splits != N_OUTER_FOLDS:
        raise ValueError(
            f"only the frozen {N_OUTER_FOLDS}-fold random split is "
            f"supported; got n_splits={n_splits}")
    return _FrozenRandomFolds(n_splits=n_splits, seed=seed)


def get_inner_folds(split_type: str, n_splits: int = N_INNER_FOLDS,
                    seed: int = SEED):
    """Return the inner splitter for tuning, on training speakers only.

    Grouped folds use a plain-numpy port of scikit-learn 1.9.0
    ``GroupKFold``; stratified folds use a plain-numpy port of
    ``StratifiedKFold(shuffle=True)``. Neither calls into the
    installed scikit-learn splitter.
    """
    if split_type == "grouped":
        return _NumpyGroupKFold(n_splits=n_splits)
    return _NumpyStratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed)


def outer_splits(split_type: str, X, y, groups) -> list:
    """Return the list of ``(train_idx, val_idx)`` outer folds.

    Parameters
    ----------
    split_type : str
        ``"grouped"`` (by speaker, the real one) or ``"random"``.
    """
    if split_type == "grouped":
        return list(get_dev_folds().split(X, y, groups=groups))
    if split_type == "random":
        return list(get_random_folds().split(X, y))
    raise ValueError(split_type)
