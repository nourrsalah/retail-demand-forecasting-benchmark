# Shared evaluation protocol — v1.0

**Status: core evaluation design locked following user review on 2026-09-17. Stage 4 implementation and validation began on 2026-09-23; final tests remain unevaluated.**

Machine-readable decisions: `configs/evaluation.json`. Model-specific checkpoints, search spaces, and resource limits must still be recorded and frozen using validation only before any final-test run.

Research question: When do pretrained time-series models provide a real advantage for business demand forecasting, and when do locally trained models remain more reliable?

## 1. Scope and information parity

Use the frozen 200-series pilot manifest. Forecast daily observed unit sales at the product/store level; this does not estimate latent demand during stockouts. Horizons: 7, 14, 28 days. Generate one 28-day forecast per origin and score its first 7/14/28 days. Never update that forecast with actuals inside the forecast window.

**Univariate benchmark (primary):** sales history only for every model (including target-derived lags/rolling features). XGBoost is a pooled local model trained on the pilot; ARIMA/ETS are fitted per series. Report this difference in cross-series information sharing explicitly. Fixed identifiers may be used to route series, but do not introduce static/category embeddings in the primary track.

**Business-feature benchmark (secondary):** sales plus price/calendar/event features where supported, reported separately from the univariate ranking. Compare models with identical covariate availability within each matched-information comparison. Unsupported feature sets must be marked unsupported, not silently replaced with another information set. A comparison of feature-rich XGBoost against sales-only foundation models must be labeled **realistic deployment comparison**, not a controlled model-family comparison. Known dates/day-of-week and deterministic calendar features may be supplied. Events/SNAP require an explicit assumption that schedules are known at the origin. Realized future sell prices are not assumed known; use only prices observed through origin, or an explicitly labeled planned-price scenario. M5 has no inventory or explicit promotion observations. Do not present event indicators as observed promotions.

Pretrained models start zero-shot with frozen checkpoints; fitting/fine-tuning, if later authorized, is a separate track. Freeze checkpoint IDs, package versions, context limits, preprocessing, and output settings before test runs. Audit each checkpoint's disclosed pretraining data for M5 overlap; unknown or possible overlap must be reported and limits clean zero-shot generalization claims.

## 2. Locked rolling-origin backtest with fixed-length history windows

An origin is the last observed day. All horizons start the next day. Initial available history begins 2011-01-29 (d_1). Train only through each origin; transformations, feature engineering, and scale denominators use only that history.

| Role | Origin | Forecast window (28 days) |
|---|---|---|
| Validation 1 | 2015-11-08 (d_1745) | 2015-11-09–2015-12-06 |
| Validation 2 | 2015-12-06 (d_1773) | 2015-12-07–2016-01-03 |
| Validation 3 | 2016-01-03 (d_1801) | 2016-01-04–2016-01-31 |
| Test 1 | 2016-01-31 (d_1829) | 2016-02-01–2016-02-28 |
| Test 2 | 2016-02-28 (d_1857) | 2016-02-29–2016-03-27 |
| Test 3 | 2016-03-27 (d_1885) | 2016-03-28–2016-04-24 |
| Test 4 | 2016-04-24 (d_1913) | 2016-04-25–2016-05-22 |

All seven origins are strictly chronological, spaced 28 days apart, and their 28-day target windows do not overlap. The last validation target is 2016-01-31; the first test target is 2016-02-01.

Choose hyperparameters using validation only, then freeze model choices, checkpoints, preprocessing, feature sets, tuning settings, and history-sensitivity configurations before any final-test forecasts or scores are produced. No test scores, plots, or series-level errors may guide these choices. Stage 1/2 integrity audits have already read the full source data; “untouched test” means excluded from model development and selection, not that these raw rows have never been loaded. At subsequent test origins, refit using newly observed past data with the frozen settings; this simulates operational updating. Earlier test-period actuals become available only after their forecast window closes and may enter the next origin’s fixed-length training window. This scheduled refit is allowed; revising model choices from test performance is not. Test results must not influence configuration or sample changes. Calendar-only dates after 2016-05-22 have no sales labels here and are not test targets.

Primary comparison: the same last **365 daily observations** for every model at every origin, inclusive of the origin day. The window slides forward with each origin; it is not expanding history.

A later, separate **history-length sensitivity experiment uses 90 / 180 / 365 days**, with the same series, origins, horizons, and univariate information set. The 365-day condition reuses the primary experiment when all settings match; do not count it as an independent replicate. The question is whether pretrained models become relatively more useful as local history decreases. Full-history experiments are outside v1.0.

Apply the same prespecified validation search space and budget at each history length; select and freeze settings separately by history length before any final-test evaluation, recording the selections. Confirm every chosen checkpoint supports each required context; do not silently truncate or pad histories to imply equal information. Mark unsupported conditions explicitly.

RMSSE scaling uses the permitted training window for each condition, so its denominator can differ across history lengths. Compare model families within each history condition using RMSSE; use MAE/WAPE on the identical targets to quantify error changes across history lengths, rather than interpreting changing-scale RMSSE as pure accuracy gains.

Global XGBoost training examples and all their labels must lie inside the permitted window. Lag warm-up rows reduce its available examples and must be reported. Multi-step features must use predictions or a training-only direct strategy, never future actual demand. Reduced-history experiments restrict every model's input, not just foundation models.

## 3. Models and tuning

### Model-family terminology

- **Naive reference baselines:** last-value and weekly seasonal-naive forecasts; reference rules, reported separately from fitted model families.
- **Locally fitted statistical models:** ARIMA and ETS, each fitted separately to an individual product/store series.
- **Locally trained machine-learning model:** XGBoost, trained as a pooled/global model across the pilot series using only the permitted local dataset and history window.
- **Pretrained time-series foundation models:** Chronos-2, TimesFM, and Moirai, evaluated initially with frozen pretrained checkpoints in zero-shot mode (no parameter fitting on the pilot).

“Local” describes the source of fitting/training data, not the hardware location or necessarily one model per series. “Per-series” versus “pooled/global” describes how fitting shares information across series. The research comparison groups ARIMA/ETS and XGBoost under **locally fitted/trained models**, while retaining their statistical/ML family distinctions in results. These terms do not change the existing information-access rules or authorize fine-tuning.

### Validation model selection

Include last-value and weekly seasonal-naive baselines alongside ARIMA, ETS, XGBoost, Chronos-2, TimesFM, and Moirai. ARIMA seasonal variants/search bounds, ETS options, XGBoost forecast strategy, checkpoint choices, and per-family tuning budgets remain review decisions. Log every candidate and validation result; no test tuning. Selection objective: first calculate the macro mean RMSSE over eligible validation series/origin cases separately for each horizon, then minimize their equally weighted mean:

`selection_score = (macro_RMSSE_7 + macro_RMSSE_14 + macro_RMSSE_28) / 3`

Each horizon receives weight **1/3**, regardless of its number of forecast days. Do not pool daily errors across horizons or weight horizons by 7/14/28. Apply the same equal-horizon aggregation to macro MAE for tie-breaking. If RMSSE is undefined for every case, use that equally weighted MAE objective and report the exception. Existing eligibility, coverage, and failure-reporting rules remain unchanged; missing horizon scores are not zero and must not cause silent redistribution of horizon weights. Apply this rule within each separately selected history-length/scenario condition; test scores never enter selection.

## 4. Point forecast metrics

Evaluate per series/origin/horizon. Use unrounded nonnegative forecasts for unit demand (clip below zero consistently); retain raw predictions and count clipped values. For distributional models, use the predictive mean where supported, documenting the extraction method. Do not silently mix medians and means under a common point-estimator label.

- **MAE:** mean absolute error in units. Report equal-series macro averages plus segment summaries; high-volume series naturally have larger absolute errors.
- **WAPE:** sum absolute errors / sum observed units, calculated by pooling the explicitly named group, not by averaging per-series percentages. If its total observed demand is zero, return NA and disclose the count; retain MAE. WAPE emphasizes volume.
- **RMSSE:** sqrt(forecast MSE / mean squared consecutive-day training differences). Use the actual permitted training history, remove leading zero-demand observations before the first positive sale, and use lag 1 (M5-style scaling). If fewer than two observations remain or the scale is zero, return NA, record why, and report eligible coverage. This is a bottom-level, unweighted RMSSE benchmark, not the official hierarchical M5 WRMSSE score.

Report scores by horizon and origin as well as pooled summaries. Aggregate only the declared complete cases, showing coverage. Never replace undefined scores with zero.

## 5. Reliability and uncertainty

**Performance consistency across series is a required result.** For each model/history/scenario/horizon and each metric MAE/RMSSE, first average each series’ errors over the same scheduled origins, requiring valid scores at every origin for the primary consistency table. Then report the median, p90, p95, standard deviation, IQR, and maximum across series, with support/exclusion counts and worst-case series IDs. Use linear-interpolated quantiles. Also report origin-specific distributions so averaging cannot hide a bad forecast period. WAPE remains a pooled group metric, not a per-series tail metric. For win rates, average each series' score over origins at each horizon, compare only cases with valid predictions from all models, and split tied wins equally (numeric tolerance 1e-8). Report excluded comparisons. RMSSE determines primary wins; MAE wins are supplementary.

Record exceptions, timeouts, missing forecasts, nonfinite values, and negative raw values. Failure denominator: scheduled model/series/origin runs (not three duplicated horizon tasks). A proposed timeout is 120 seconds per per-series fit+forecast or 30 minutes per pooled/batched origin job; review after a validation-only timing smoke test, then freeze. Do not hide failures behind fallback predictions. If operational fallbacks are explored, score and label that composite system separately.

Present complete-case accuracy together with per-model coverage/failure rate; a model with missing hard cases cannot claim a general win from its successful subset alone. Use paired bootstrap confidence intervals over series, retaining all origins within each resampled series. State that shared stores/products create dependence and pilot intervals are descriptive, not population-wide guarantees.

## 6. Segmentation and history experiments

Freeze Stage 2 demand/volume labels from the first 730 days; use the documented zero-fraction/CV thresholds and store/category volume thirds. These are overlapping axes: demand type is exclusive, volume is separate. Report state, store, category, and segment counts. The pilot excludes all-zero sampling histories and is diversity-balanced; conclusions do not cover all new products or the full population automatically.

A proposed exploratory weekly-seasonality flag: on the same 730-day prefix, compute lag-7 autocorrelation after subtracting a trailing 28-day mean; flag >= 0.3 with >= 365 usable residual pairs and nonzero residual variance, otherwise unclassified. This is a descriptive heuristic, not proof of seasonality; freeze before model results. Do not infer thresholds after seeing wins.

History length (90/180/365) is an experimental axis, not a label derived from future data. Covariate gains are within-model secondary comparisons and must not be conflated with pretrained-versus-local differences.

## 7. Runtime, compute, and business practicality

Use the same recorded host where feasible. Log CPU/GPU type, RAM/VRAM, OS, package versions, precision, threads, batch size, and random seeds. Separate dependency/setup and checkpoint download time, model loading, fitting/tuning, feature preparation, and forecast inference. Time synchronized GPU operations; mark unavailable measurements as NA.

Capture peak process RSS (including child processes where applicable) and peak device memory per isolated run. Separate cold start from warmed inference; repeat warmed timing three times and report median and range. Report both total pilot wall time and throughput; do not pretend pooled-model time is a measured per-series fit time. Zero-shot models still incur loading/inference cost. Record retraining policy, failed setup steps, covariate support, deployment dependencies, and license constraints from pinned model documentation.

No assumed profit or inventory-cost savings from accuracy alone. Any business cost scenario must state stockout/holding-cost assumptions; M5 lacks inventory observations for causal operational conclusions.

## 8. Reproducibility and output contracts

Seed 42 where supported; record nondeterministic operations. Save raw-file hashes, pilot manifest, protocol version/hash, config, code revision (when a project repository exists), environment, model checkpoint/revision, and run IDs. Store timestamps and status/error logs. Avoid committing model weights and raw/processed data.

Forecast columns: `run_id, model, model_version, scenario, history_window, series_id, forecast_origin, horizon, lead, date, y_true, y_pred_raw, y_pred, status`. Each horizon contains leads 1..horizon; overlapping predictions must be identical across horizons. Unique key: `(run_id, model, scenario, history_window, series_id, forecast_origin, horizon, lead)`.

Metric columns: matching run/model/scenario/series/origin/horizon identifiers, `metric, value, valid, reason, n_observations`. Resource logs are separate at their measured granularity. Full forecasts go under `results/forecasts/`; comparison tables under `results/metrics/`; figures under `results/figures/`.

## Change control and implementation boundary

The reviewed core design is locked: horizons 7/14/28; seven chronological rolling origins; primary 365-day history; later 90/180/365 sensitivity; MAE/WAPE/RMSSE; median/tail errors, win rates and failures; runtime/memory/hardware/setup effort; and separate covariate experiments. The pilot remains 200 series across all stores/states/categories, selected from the first 730 days, excluding cold starts.

Stage 4 statistical execution details are now recorded in `configs/stage4.json` and `docs/stage4.md`, under the unchanged core protocol. Remaining execution details (specific checkpoints, model search spaces, exact budgets/timeouts, and optional exploratory seasonality) must be finalized and logged using validation only before final-test access. These details cannot alter the locked core design without a versioned amendment. Any later protocol changes require a new version and rationale; test-informed changes require fresh holdouts or explicit exploratory labeling.

### v1.0 review changes

- Retained standardized 365-day history and specified 90/180/365 sensitivity instead of the draft’s 90/full-history extension.
- Made chronological separation, frozen choices, and allowed scheduled refits explicit.
- Named univariate and business-feature tracks, with a separate label for unequal-information deployment comparisons.
- Made cross-series median/p90/p95 reporting mandatory and defined aggregation and missing-score handling.

### v1.0 terminology and selection clarification

Explicitly named the model families and clarified equal (1/3 each) validation selection weights for the 7/14/28-day horizons, including MAE tie-breaking/fallback. All other locked design decisions remain unchanged.
