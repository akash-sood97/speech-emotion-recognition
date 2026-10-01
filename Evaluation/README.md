# Evaluation

Three notebooks that judge the model ladder beyond a single score. They compare
the classical rung, the log-mel CNN and the frozen speech encoders on the same
predictions, turn confidence into a review flag, and hold the last exam on the
six speakers nobody has looked at. Every claim is a numbered hypothesis
(H31 onwards) with a verdict driven by printed numbers.

## 1 · What you need

- Fold assignment is stored in `shared/frozen_split.json`, so results do not depend on the scikit-learn version.
- **Python 3.13**
- The packages in `requirements.txt` (no torch required for the default run)
- Finished **`EDA and Feature engineering/`** and **`Modelling/`** sections
- The audio dataset beside this folder (course-supplied, not included; see `docs/HOW_TO_RUN.md`). Notebooks find the project root by looking for `data/`; the default
  path does not read audio.

```
your folder/
├── data/
│   ├── Person01/
│   └── … Person24/
└── Group-10/
    ├── shared/
    ├── EDA and Feature engineering/
    ├── Modelling/
    └── Evaluation/                    ← this folder
```

Notebooks run without `cache/`. Held-out predictions and `recipe_check.csv`
are shipped under `outputs/`.

## 2 · Set up (once)

```bash
pip install -r requirements.txt
```

## 3 · Run

```bash
jupyter nbconvert --to notebook --execute --inplace 06_Ladder_on_development_speakers.ipynb
jupyter nbconvert --to notebook --execute --inplace 07_Flag_rate_and_threshold.ipynb
jupyter nbconvert --to notebook --execute --inplace 08_Final_test.ipynb
```

| Notebook | Default | RUN_HEAVY |
|---|---|---|
| `06_…` | Recompute ladder / H31–H34 from Modelling preds | none |
| `07_…` | Flag-rate sweep / H35–H36 | none |
| `08_Final_test.ipynb` | Recipe check from `recipe_check.csv`; held-out metrics from shipped `*_heldout.csv` (`OPEN_HELD_OUT = True`) | Rebuilds the development recipe check (needs WavLM embeddings under `cache/ssl_embeddings/`, built by Modelling notebook 05). **Never overwrites** the saved held-out predictions |

## 4 · What you get

Held-out WavLM macro-F1 **0.7596** (`outputs/tables/heldout_metrics.csv`),
figures F10 / F10b–d / F14 / F15, and summary tables for H31–H39.

## 5 · If something fails

| Message | Fix |
|---|---|
| Missing Modelling preds | run Modelling notebooks 03–05 first (default path) |
| Missing emb_wavlm.npy on RUN_HEAVY | run Modelling `05` with `RUN_HEAVY = True`, or keep `RUN_HEAVY = False` |
