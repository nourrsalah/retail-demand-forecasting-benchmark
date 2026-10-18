# Stage 6 validation results — pretrained models

Run: `20261006T004217634146Z-stage6-validation`. **Validation only; final test periods remain unevaluated.**

Three prespecified frozen checkpoints, 200 series, three origins, 365 observed daily sales values per forecast. No fitting, future labels or business covariates. 0 failures out of 1800 scheduled model/series/origin cases. Full wall time including three warm repeats: 567.7 seconds.

## Native-estimator comparison

The following table explicitly mixes native point estimators. Chronos-2 supplies a **median**, while TimesFM supplies its mean head and Moirai supplies a 200-sample mean. It is an operational comparison, not a controlled all-means ranking. Each number gives equal 1/3 weight to the 7/14/28-day aggregates. WAPE is pooled within each horizon.

| model | point_estimator | MAE | RMSSE | WAPE |
| --- | --- | --- | --- | --- |
| ARIMA | local_point_forecast | 1.0655 | 0.6609 | 0.8286 |
| Chronos-2 | native_median | 0.9617 | 0.6706 | 0.7478 |
| ETS | local_point_forecast | 1.0661 | 0.6612 | 0.8291 |
| Moirai | sample_mean_200 | 1.1304 | 0.7039 | 0.8791 |
| TimesFM | native_mean | 1.0311 | 0.6501 | 0.8019 |
| XGBoost | local_point_forecast | 1.0820 | 0.6826 | 0.8413 |
| last_value | local_point_forecast | 1.4200 | 0.8737 | 1.1046 |
| weekly_seasonal_naive | local_point_forecast | 1.3109 | 0.8876 | 1.0195 |

Common RMSSE-eligible support per horizon: 599–599 out of 600. Per-model coverage and failures are retained in `combined_coverage.csv`. A separate `comparison_mean_only.csv` excludes Chronos-2; its win rates are in `win_rates_mean_only.csv`. The naive forecasts remain reference rules. Existing ARIMA/ETS/XGBoost choices were reused, not retuned.

## Runtime and memory

| candidate | failed_origins | first_inference_seconds | peak_worker_RSS_MiB |
| --- | --- | --- | --- |
| chronos2 | 0 | 14.84 | 765.7 |
| moirai11small | 0 | 91.09 | 444.8 |
| timesfm25 | 0 | 32.59 | 1300.7 |

All models run sequentially on the recorded CPU host. First inference is summed across origins; loading and three warmed repeats are separate in `resources.csv`. RSS includes the full worker runtime, sampled every 10 ms; brief peaks can be missed. Checkpoint downloads and model initialization are not training time. Moirai's selected checkpoint has a noncommercial license.

## Limits and saved evidence

- `accuracy_by_horizon.csv`, `case_scores.csv`, `metrics.csv`: per-model accuracy, eligibility and failures.
- `reliability_mean_only.csv` / `reliability_native_estimators.csv`: across-series median, p90, p95, spread and maximum; complete origins required.
- `origin_distributions.csv`: origin-level distributions.
- `candidate_selection.csv`: equal-horizon validation objective, one prespecified configuration per model.
- `run.json`, checkpoint lock and environment files: data/source hashes, pinned revisions, settings and hardware.
- `results/forecasts/20261006T004217634146Z-stage6-validation/forecasts.csv.gz`: raw and clipped forecasts.

Moirai’s published family pretraining corpus includes M5. No M5-disjoint pretraining guarantee has been established for these checkpoints; see `docs/pretraining_audit.md`. These scores cannot demonstrate clean unseen-data generalization or a universal model-family winner. Pilot selection excludes cold starts. History sensitivity, business features, uncertainty intervals and the frozen final evaluation remain later work.
