# Modelling

Three notebooks that climb a model ladder on the speech-emotion audio dataset:
classical classifiers on the 512 acoustic features, a small log-mel CNN, then
frozen speech encoders (wav2vec2 / HuBERT / WavLM) and a fusion check. Every
claim is a numbered hypothesis (H20–H30) with a verdict driven by printed numbers.

## 1 · What you need

- Fold assignment is stored in `shared/frozen_split.json`, so results do not depend on the scikit-learn version.
- **Python 3.13** (the environment that ran these notebooks)
- The packages in `requirements.txt`
- The audio dataset beside this folder (course-supplied, not included; see `docs/HOW_TO_RUN.md`)
- The finished **`EDA and Feature engineering/`** section (for
  `features.parquet` and the label index)

```
your folder/
├── data/
│   ├── Person01/
│   └── … Person24/
└── speech-emotion-recognition/
    ├── shared/
    ├── EDA and Feature engineering/
    └── Modelling/                     ← this folder
```

Notebooks run with `RUN_HEAVY = False` using only the shipped `outputs/`
(predictions, tables, figures). No `cache/` folder is required for the default
run.

## 2 · Set up (once)

Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
```

## 3 · Run

Run the three notebooks **in order** (after `01_EDA` and `02_Feature_engineering`):

```bash
jupyter nbconvert --to notebook --execute --inplace 03_Classical_models.ipynb
jupyter nbconvert --to notebook --execute --inplace 04_CNN.ipynb
jupyter nbconvert --to notebook --execute --inplace 05_Pretrained_encoders.ipynb
```

| Notebook | Live path (default) | RUN_HEAVY cells |
|---|---|---|
| `03_Classical_models.ipynb` | Recompute classical metrics from fold preds; H20–H23 | Retrain logreg/svc/rf/gbm and logreg-spkz via `H.run_cv` (audio not needed; needs `features.parquet`; about 10–20 min) |
| `04_CNN.ipynb` | Rebuild 3-seed ensemble from `preds/seeds`; H24–H25 | Mel cache from waveforms, SmallCNN training, Grad-CAM F12 (needs audio + torch; GPU recommended; hours) |
| `05_Pretrained_encoders.ipynb` | Rebuild fusion; ladder; selection rule; redraw F09/F11/F13; H26–H30 | Encoder download + embedding extraction to `cache/ssl_embeddings/`, F11b layer profile, head fits, F11 importance (needs audio + torch + transformers; GPU recommended; hours) |

Set `RUN_HEAVY = True` only when you intend to rebuild those artefacts. Default
is `False`.

## 4 · What you get

| Folder | Files |
|---|---|
| `outputs/tables/` | `model_comparison.csv`, `selection.json` (**wavlm** 0.7535 ± 0.047), baselines, F11 / F11b tables |
| `outputs/preds/` | Per-fold predictions for every rung |
| `outputs/figures/` | F09, F11, F11b, F12, F13 |

**WavLM head (what was run):** layer ∈ {0,…,12} and C ∈ {0.01, 0.1, 1.0}
by inner speaker-grouped CV; `class_weight="balanced"`.

## 5 · If something fails

| Message | Fix |
|---|---|
| `Could not find the audio dataset` | place `data/` beside `speech-emotion-recognition/` |
| `features.parquet` missing | run `02_Feature_engineering.ipynb` first |
| `ModuleNotFoundError` | `pip install -r requirements.txt` in the Jupyter environment |
| Fold prediction file missing | keep `RUN_HEAVY = False` so shipped `outputs/preds/` are used |
