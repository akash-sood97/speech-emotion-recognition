# EDA and Feature engineering

Two notebooks that explore the speech-emotion audio dataset and turn every clip into a table of 512 features.
Every claim is written as a hypothesis, tested on the data, and ends in a verdict and the decision it drove.

## 1 · What you need

- Fold assignment is stored in `shared/frozen_split.json`, so results do not depend on the scikit-learn version.
- The audio is course-supplied and not included; see `docs/HOW_TO_RUN.md`.
- Notebooks run without `cache/`; rebuildable intermediates are optional.

- **Python 3.13** (other 3.11+ versions will likely work, but 3.13 is what was tested)
- About **1 GB** of free disk space
- The audio dataset folder **`data/`** (`Person01` … `Person24`, 1,440 `.wav` files), placed **beside**
  `speech-emotion-recognition/`:

```
your folder/
├── data/
│   ├── Person01/
│   └── … Person24/
└── speech-emotion-recognition/
    ├── shared/                        helper code used by every notebook
    └── EDA and Feature engineering/   ← this folder
```

## 2 · Set up (once)

Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
```

## 3 · Run

Run the two notebooks **in order**:

```bash
jupyter nbconvert --to notebook --execute --inplace 01_EDA.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_Feature_engineering.ipynb
```

Or start Jupyter with `jupyter notebook`, open each notebook, and choose **Run → Run All Cells**.

| Notebook | What it does | Time (8-core laptop) |
|---|---|---|
| `01_EDA.ipynb` | Checks the files, tests H1–H11, and fixes the preprocessing and the speaker split | about 20 s |
| `02_Feature_engineering.ipynb` | Builds or loads the 512 features, tests H12–H19, and fixes the feature recipe | about 20 s |

The first run also builds a waveform cache in `speech-emotion-recognition/cache/` (a few seconds; about 160 MB).

## 4 · What you get

| Folder | Files |
|---|---|
| `outputs/tables/` | `label_index.csv` (one row per clip), `audit_report.csv`, `dataset_profile.json` (every dataset figure quoted anywhere), `acoustics.csv`, `emotion_profile.csv`, `window_test.csv`, `eta_squared.csv`, `eta_by_group.csv`, `redundancy.json`, `silhouette.json`, `confusion_pairs.csv`, `duplicate_check.json` |
| `outputs/features/` | `features.parquet` (1,440 clips × 512 features), `feature_manifest.json` (exactly how it was built) |
| `outputs/figures/` | F01–F07c, the charts used in the slides and report |

## 5 · Full rebuild (optional)

By default, `02_Feature_engineering` loads the saved feature table and the saved window-test results.

To rebuild everything from the audio, set `RUN_HEAVY = True` in the first code cell of `02_Feature_engineering` and
run it again. This re-extracts all features and re-scores the 12 window-test configurations, and takes about
3½ minutes on 8 cores. The results are identical to the saved files.

To rebuild the pitch table, delete `outputs/tables/acoustics.csv`; `01_EDA` then rebuilds it (15 s to 5 minutes).

## 6 · If something fails

| Message | Fix |
|---|---|
| `Could not find the audio dataset` | `data/` must sit beside `speech-emotion-recognition/`, as in the tree in §1 |
| `ModuleNotFoundError` (e.g. `librosa`) | run `pip install -r requirements.txt` in the same Python environment Jupyter uses |
| `label_index.csv not found` | run `01_EDA.ipynb` before `02_Feature_engineering.ipynb` |
| Numbers differ slightly from the saved outputs | check the package versions against `requirements.txt` |
