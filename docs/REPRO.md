# Reproduction check

- Date: 1 October 2026
- Python: 3.13.9 (fresh virtual environment, `pip install -r requirements.txt` only)
- Setup: repo copied to a scratch folder, course audio linked as `data/` beside it, `RUN_HEAVY = False` (default), notebooks run in order with `jupyter nbconvert --to notebook --execute --inplace`.

## Run time per notebook

| Notebook | Seconds |
|---|---|
| 01_EDA | 38 |
| 02_Feature_engineering | 23 |
| 03_Classical_models | 4 |
| 04_CNN | 9 |
| 05_Pretrained_encoders | 3 |
| 06_Ladder_on_development_speakers | 4 |
| 07_Flag_rate_and_threshold | 3 |
| 08_Final_test | 7 |
| Total | 91 |

Notebook 01 builds a local waveform cache from the audio on its first run; no pre-built cache or model download was needed.

## Comparison with the committed outputs

`Evaluation/outputs/tables/heldout_metrics.csv` and `Evaluation/outputs/tables/summary_08.csv` from the fresh run are byte-identical to the committed files.

| model | macro_f1 |
|---|---|
| wavlm | 0.7596 (95% CI 0.6527 to 0.8519) |
| logreg | 0.5775 |
| majority | 0.0294 |
