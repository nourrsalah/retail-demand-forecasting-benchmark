# Stage 6 — frozen pretrained models

This stage implements the **pretrained time-series foundation** family under protocol v1.0. The core protocol and pilot are unchanged. It exposes validation only, with no final-test switch.

## Prespecified checkpoint budget

One checkpoint and one configuration per family; no accuracy-driven search. Choices target the available 8 GB CPU host. Exact immutable revisions, downloaded-file SHA-256 hashes and download times are stored in `configs/stage6_checkpoints.json`.

| Model | Checkpoint | Point extraction | License |
| --- | --- | --- | --- |
| Chronos-2 | amazon/chronos-2 | Native 0.5 quantile (median) | Apache-2.0 |
| TimesFM 2.5 | google/timesfm-2.5-200m-pytorch | Native point head (mean) | Apache-2.0 |
| Moirai 1.1 small | Salesforce/moirai-1.1-R-small | Mean of 200 seeded predictive samples | CC-BY-NC-4.0 |

Chronos-2's `predict_quantiles` returns a median in its output named `mean`. It is explicitly labeled `native_median`, excluded from the mean-only comparison, and included in a separately labeled native-estimator table. No artificial distributional mean is extrapolated from a finite quantile grid. TimesFM's first returned point array is also the median; this adapter deliberately extracts channel 0 (the mean head) from its full forecast output. Moirai's mean is a Monte Carlo estimate; seed 42 is reset for repeated timings. Its noncommercial license matters to business deployment.

## Input and execution

Each model sees exactly 365 observed sales values ending at the origin, no future labels, calendar, price, identifiers as predictors, or fine-tuning. Chronos cross-learning is disabled. TimesFM's context allocation is 384 (32-step patch alignment); its internal masked padding does not add observed history. Its internal horizon allocation is 128; only the first 28 output days are used. Moirai uses fixed patch size 16, not an automatically selected patch size.

Generate one 28-day path and score the same first 7/14/28 values. Common nonnegative clipping happens outside adapters, with raw values retained. Equal 1/3 horizon aggregation is reused. Invalid/missing forecasts stay failures; no fallback predictions are substituted. RMSSE eligibility remains unchanged.

Models run sequentially in fresh CPU processes, float32, four Torch threads, batch size four. Dependency installation is isolated in `.venv-stage6`; Stage 1–5 dependencies are preserved. Three seeded warm repeats accompany first inference. Loading, inference, warm median/range, sampled worker peak RSS and overall wall time are separate. RSS is sampled every 10 ms and includes the model runtime; shared model peak is repeated on origin rows and must not be summed. No GPU is used. Dependency installation elapsed time was not measured precisely and is NA; checkpoint download times are recorded separately. The worker budget is 30 minutes per origin plus model loading, bounded externally across the run.

## Reproduce

From the project root:

```sh
.venv/bin/python -m venv .venv-stage6
.venv-stage6/bin/python -m pip install -r requirements/stage6.lock.txt
.venv-stage6/bin/python -m src.utils.prepare_stage6
.venv/bin/python -m src.evaluation.stage6 --smoke
.venv/bin/python -m src.evaluation.stage6
```

The download helper reuses pinned revisions, checks file hashes, keeps 2 GiB free disk headroom, and does not delete files. Model execution is offline from the verified local snapshots. Weights, environments and the worker inputs are ignored by Git. `--model chronos2|timesfm25|moirai11small` permits isolated diagnosis; a comparison report requires a full three-model run.

The smoke subset is the first sorted series in each frozen demand segment, at the first validation origin. It checks compatibility, not model selection. Full validation covers 200 series × 3 origins per model. Failures and coverage are included even if a library or checkpoint fails to load.

## Interpretation boundary

See `pretraining_audit.md`: no clean M5-disjoint pretraining claim is established. Zero-shot here means no fitting on this project's pilot, not proven absence of M5 in pretraining. Accuracy claims are validation-only. History sensitivity, business covariates, paired uncertainty analysis and final tests remain later experiments.
