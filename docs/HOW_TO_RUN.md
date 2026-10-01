# Group 10 · Speech Emotion Recognition: code package

This package predicts the emotion in a short speech clip (8 emotions) and turns the prediction into a review flag: it
suggests which calls a quality-assurance reviewer should hear first. It holds eight notebooks in three sections, the
helper code they share, and every table and figure used in the slides and the report.

**Headline result.** A frozen WavLM speech encoder with a logistic-regression head reaches a macro-F1 of
**0.7596** (95% interval 0.6527–0.8519) on six speakers that no step of model building ever saw. Logistic regression
on 512 acoustic features reaches 0.5775, and always predicting one emotion reaches 0.0294. Source:
`Evaluation/outputs/tables/heldout_metrics.csv`.

## 1 · What is in the folder

```
your folder/
├── data/                               the audio dataset (download it, see §2)
│   ├── Person01/ … Person24/           1,440 .wav files
└── speech-emotion-recognition/                           this package
    ├── README.md                       this file
    ├── shared/                         helper code every notebook imports
    │   ├── audio_utils.py              audio loader, preprocessing, paths, chart style
    │   ├── features.py                 the 512 acoustic features
    │   ├── speaker_split.py            the speaker split and cross-validation folds
    │   ├── cv_harness.py               speaker-grouped cross-validation and tuning
    │   ├── redraw_modelling_figures.py redraws figures F09, F11, F13
    │   └── frozen_split.json           which speakers are development and which are held out
    ├── EDA and Feature engineering/    notebooks 01–02, hypotheses H1–H19
    ├── Modelling/                      notebooks 03–05, hypotheses H20–H30
    └── Evaluation/                     notebooks 06–08, hypotheses H31–H39
```

Each section folder has its own `README.md`, `requirements.txt`, notebooks and `outputs/` (tables, figures,
predictions). Every notebook is saved with its outputs, so it can be read without running anything.

Every testable claim in the notebooks is a numbered hypothesis (H1–H39). Each one is followed by its evidence, a
verdict and the decision it drove, and each notebook ends with a summary table.

## 2 · What you need

- **Python 3.13** (the version that ran the notebooks; other 3.11+ versions will likely work)
- About **1 GB** of free disk space. No GPU is needed for the default run.
- **The audio.** The audio (1,440 acted speech clips from 24 speakers) was supplied by the course and is not included in this repo. Every notebook is saved with its outputs, so the work can be read without it. To re-run, place the course audio as `data/Person01` … `data/Person24` beside this folder.

## 3 · Set up (once)

Open a terminal in `speech-emotion-recognition/` and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r "EDA and Feature engineering/requirements.txt" -r Modelling/requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead. The Modelling file also covers everything Evaluation
needs.

## 4 · Run, in this order

Each notebook reads what the previous ones wrote, so run them in order. From `speech-emotion-recognition/`:

```bash
cd "EDA and Feature engineering"
jupyter nbconvert --to notebook --execute --inplace 01_EDA.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_Feature_engineering.ipynb
cd ../Modelling
jupyter nbconvert --to notebook --execute --inplace 03_Classical_models.ipynb
jupyter nbconvert --to notebook --execute --inplace 04_CNN.ipynb
jupyter nbconvert --to notebook --execute --inplace 05_Pretrained_encoders.ipynb
cd ../Evaluation
jupyter nbconvert --to notebook --execute --inplace 06_Ladder_on_development_speakers.ipynb
jupyter nbconvert --to notebook --execute --inplace 07_Flag_rate_and_threshold.ipynb
jupyter nbconvert --to notebook --execute --inplace 08_Final_test.ipynb
```

Or start Jupyter with `jupyter notebook`, open each notebook in turn, and choose **Run → Run All Cells**.

| # | Notebook | Question it answers |
|---|---|---|
| 01 | `01_EDA` | Is the audio dataset clean, and how should it be split and preprocessed? |
| 02 | `02_Feature_engineering` | Which 512 acoustic features describe each clip, and over which part of the clip? |
| 03 | `03_Classical_models` | Do tuned classical models on those features beat guessing? |
| 04 | `04_CNN` | Does a small CNN on spectrograms beat them? |
| 05 | `05_Pretrained_encoders` | Do frozen pre-trained speech encoders do better, and which model is selected? |
| 06 | `06_Ladder_on_development_speakers` | Where does each model go right and wrong? |
| 07 | `07_Flag_rate_and_threshold` | How should confidence become a review flag? |
| 08 | `08_Final_test` | Does the selected model hold up on six speakers it has never heard? |

**Default run versus full rebuild.** Every notebook has a switch `RUN_HEAVY = False` in its first code cell. With
`False`, the slow steps load their saved results from `outputs/` and print `[cache hit]`:

- classical retraining;
- CNN training;
- encoder downloads and embeddings.

Set it to `True` only to rebuild those results from the audio. That takes from minutes (classical models) to hours
(CNN and encoders; a GPU is recommended). It needs the `torch` and `transformers` packages listed in
`Modelling/requirements.txt`. The per-section READMEs give the details.

## 5 · What you get

| Where | What |
|---|---|
| `EDA and Feature engineering/outputs/` | Dataset profile, audit tables, the 512-feature table, figures F01–F07c |
| `Modelling/outputs/` | Per-fold predictions of every model, `model_comparison.csv`, `selection.json`, figures F09–F13 |
| `Evaluation/outputs/` | Error analysis, the flag rule (`flag_operating_point.json`), held-out results (`heldout_metrics.csv`), figures F10, F10b–d, F14, F15 |

## 6 · How the results were kept honest

- **Speakers never cross the split.** 18 speakers (1,080 clips) are used for development and 6 speakers (360 clips)
  are held out. All 60 clips of a speaker stay on one side. The split is drawn from seed 42 and checked against
  `shared/frozen_split.json` on every load.
- **Cross-validation is speaker-grouped.** All tuning and model choice use it on the development speakers. Scalers are
  fitted inside each fold.
- **The model was chosen before the test.** `Evaluation/model_choice.md` fixed the selected model and the pass bars
  before any held-out clip was scored. `08_Final_test` scored the held-out speakers once and stored the predictions. A
  re-run loads them and never overwrites them.
- **Pre-trained encoders are frozen.** The checkpoints are public (`microsoft/wavlm-base-plus`,
  `facebook/wav2vec2-base`, `facebook/hubert-base-ls960`, pinned by revision in notebook 05). They are not fine-tuned
  for emotion; only the emotion head is trained here.

## 7 · If something fails

| Message | Fix |
|---|---|
| `Could not find the audio dataset` | `data/` must sit beside `speech-emotion-recognition/`, as in the tree in §1 |
| `ModuleNotFoundError` | Run the `pip install` line in §3 in the same environment Jupyter uses |
| `label_index.csv not found` or `features.parquet` missing | Run the notebooks in the order of §4 |
| A prediction or embedding file is missing | Keep `RUN_HEAVY = False`, so the saved `outputs/` are used |
| Numbers differ slightly from the saved outputs | Check the package versions against the `requirements.txt` files |
