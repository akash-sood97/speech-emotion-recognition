# Model choice

**Written 28 September 2026, before any held-out clip has been scored.**
This note fixed what would be scored and what would count as support, so the
held-out result could not move the goalposts.

**G4 status (appended 28 Sep 2026):** the exam was opened once in
`08_Final_test.ipynb`. Held-out WavLM macro-F1 = **0.7596** (see
`outputs/tables/heldout_metrics.csv` and `summary_08.csv`). The bars below
were **not** changed after the result.

## The model

The selected model is a **frozen WavLM encoder with a logistic-regression
head**, named by `Modelling/outputs/tables/selection.json`:

- rule: highest mean speaker-grouped macro-F1 on the development speakers;
  candidates within one standard deviation of the leader are tied, and a tie
  goes to the simplest model;
- result: WavLM, macro-F1 0.7535 ± 0.047 (six-fold speaker-grouped CV, 1,080
  development clips); wav2vec2 and the fusion model sit inside the tie band
  and lose on simplicity.

The recipe is fixed as follows, and `08_Final_test` checks that it reproduces
the saved development predictions before it scores anything else:

| Part | Setting |
|---|---|
| Encoder | `microsoft/wavlm-base-plus`, frozen, clip embedding mean-pooled over time, cached in `cache/ssl_embeddings/` |
| Layer | one of 13 hidden layers (0 to 12), chosen by inner speaker-grouped CV on the development clips |
| Head | standardise, then logistic regression with balanced class weights |
| Regularisation C | one of 0.01, 0.1, 1.0, chosen together with the layer by the same inner CV |
| Training rows | all 1,080 development clips, no held-out clip |

**Reference rung, reported for context and not selectable:** logistic
regression on the 512 acoustic features (standardised, balanced class
weights, C from 0.01, 0.03, 0.1, 0.3, 1.0 by inner speaker-grouped CV),
trained on the same 1,080 clips. The log-mel CNN is not re-scored on the
held-out speakers: it would have to be retrained, and its development result
already places it below the classical rung.

**Not eligible as the headline:** the per-speaker z-scored variants (they need
calibration audio from each caller), and the fusion model.

## What will be scored, once

The 360 clips of the six held-out speakers, by the model above, fitted once on
the 1,080 development clips. No setting is changed after the held-out result
is seen, and the result is reported whatever it is.

Primary number: **macro-F1 over the eight emotions.** Also reported: accuracy,
balanced accuracy, recall for each emotion, macro-F1 for each held-out
speaker, and a 95% interval that resamples speakers (six is a small number,
so the interval is wide).

## Hypotheses and their bars, fixed in advance

| Hypothesis | Counts as supported when |
|---|---|
| H37, the selected model keeps its development performance on unseen speakers | held-out macro-F1 is inside the development band, 0.7535 ± 0.047 (0.7065 to 0.8005); below the band is not supported; above the band is supported and is noted as a result better than expected |
| H38, the gain of WavLM over the classical rung holds on unseen speakers | WavLM has the higher held-out macro-F1 and is ahead on at least 5 of the 6 speakers |
| H39, the flag rule fixed on development speakers keeps its behaviour | at the fixed threshold the flag rate is between 25% and 35%, and capture of negative clips is not more than 10 points below the development capture (at least 44.2%) |

The flag rule is the one in `07_Flag_rate_and_threshold`: flag a clip when the
predicted emotion is angry, fearful, sad or disgust and the confidence is at
least the threshold. The threshold is **0.827**, the value that flagged 30% of
the development clips, stored in `outputs/tables/flag_operating_point.json`
(development capture 54.2%, precision 96.3%). It is not re-tuned on held-out
clips.

## What this does not settle

Six held-out speakers are a small sample: a single unusual voice can move the
held-out score by several points, as speaker 13 does on the development
speakers. The held-out number is one estimate for new speakers, next to the
development estimate, and not a replacement for it.
