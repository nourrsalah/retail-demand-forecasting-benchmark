# Stage 7 — final shared-protocol evaluation

**Final test complete.** All model and experiment choices were frozen before any final-test forecasts. The freeze is `configs/stage7_freeze.json` (2026-10-11T19:30:18.441043+00:00); its source, configuration, checkpoint and validation evidence hashes still match.

The benchmark uses the original 200 pilot series, four final origins (2016-01-31, 2016-02-28, 2016-03-27, 2016-04-24), sales histories of 90/180/365 days, and 7/14/28-day forecasts weighted equally. Local models refit using only the available history at each origin; frozen foundation models receive exactly that observed history. The separate 365-day business-feature condition is a within-XGBoost comparison.

## Primary 365-day final results

| model | MAE | RMSSE | WAPE |
| --- | --- | --- | --- |
| TimesFM | 1.0751 | 0.6914 | 0.7626 |
| ARIMA | 1.1050 | 0.7019 | 0.7838 |
| ETS | 1.1073 | 0.7029 | 0.7855 |
| XGBoost | 1.1019 | 0.7203 | 0.7817 |
| Moirai | 1.1453 | 0.7245 | 0.8124 |
| Chronos-2 | 1.0313 | 0.7291 | 0.7315 |
| last_value | 1.3791 | 0.9097 | 0.9783 |
| weekly_seasonal_naive | 1.3184 | 0.9212 | 0.9352 |

Lower is better. These native-estimator results compare Chronos-2's median with the other documented point estimates. The separate mean-only tables exclude Chronos-2. Common successful RMSSE-eligible cases are used across models within each track; coverage and failures are preserved separately. WAPE pools errors and demand within each horizon, then averages horizon values equally.

## History sensitivity

| model | history_days | MAE | WAPE |
| --- | --- | --- | --- |
| ARIMA | 90 | 1.1161 | 0.7778 |
| ARIMA | 180 | 1.1151 | 0.7771 |
| ARIMA | 365 | 1.1222 | 0.7821 |
| Chronos-2 | 90 | 1.0613 | 0.7396 |
| Chronos-2 | 180 | 1.0481 | 0.7304 |
| Chronos-2 | 365 | 1.0446 | 0.7280 |
| ETS | 90 | 1.1173 | 0.7786 |
| ETS | 180 | 1.1178 | 0.7790 |
| ETS | 365 | 1.1246 | 0.7837 |
| Moirai | 90 | 1.1939 | 0.8320 |
| Moirai | 180 | 1.1797 | 0.8221 |
| Moirai | 365 | 1.1540 | 0.8042 |
| TimesFM | 90 | 1.1026 | 0.7684 |
| TimesFM | 180 | 1.0915 | 0.7606 |
| TimesFM | 365 | 1.0896 | 0.7593 |
| XGBoost | 90 | 1.1397 | 0.7942 |
| XGBoost | 180 | 1.1183 | 0.7793 |
| XGBoost | 365 | 1.1168 | 0.7782 |
| last_value | 90 | 1.4036 | 0.9782 |
| last_value | 180 | 1.4036 | 0.9782 |
| last_value | 365 | 1.4036 | 0.9782 |
| weekly_seasonal_naive | 90 | 1.3414 | 0.9348 |
| weekly_seasonal_naive | 180 | 1.3414 | 0.9348 |
| weekly_seasonal_naive | 365 | 1.3414 | 0.9348 |

This table uses identical successful series/origin targets across all eight methods and all three history lengths: 780–780 cases per horizon out of 800. MAE and pooled WAPE are averaged with equal horizon weights. Zero-scale histories remain eligible for these unscaled metrics. Failures excluded from this matched comparison remain in the coverage table.

RMSSE has a history-specific denominator and is used only for model comparisons within a history. The separate per-history summaries can use different RMSSE-eligible support; they are not the basis for this shared-target sensitivity table. See `history_identical_targets_*.csv`.

## Separate business-feature experiment

| model | MAE | RMSSE | WAPE |
| --- | --- | --- | --- |
| XGBoost | 1.1019 | 0.7203 | 0.7817 |
| XGBoost + business features | 1.1144 | 0.7161 | 0.7905 |

This compares XGBoost with and without known-calendar and historical-price information. Events/SNAP are assumed scheduled and known at the origin; future prices are carried forward from the last observed price. No realized future prices or target labels enter forecasting. Other covariate adapters were not implemented in this experiment; this is not a family-level information-parity ranking.

## Coverage and evidence

| condition | model | scheduled_jobs | failures |
| --- | --- | --- | --- |
| univariate_365 | ARIMA | 800 | 0 |
| univariate_365 | Chronos-2 | 800 | 0 |
| univariate_365 | ETS | 800 | 0 |
| univariate_365 | Moirai | 800 | 0 |
| univariate_365 | TimesFM | 800 | 0 |
| univariate_365 | XGBoost | 800 | 0 |
| univariate_365 | last_value | 800 | 0 |
| univariate_365 | weekly_seasonal_naive | 800 | 0 |
| univariate_90 | ARIMA | 800 | 20 |
| univariate_90 | Chronos-2 | 800 | 0 |
| univariate_90 | ETS | 800 | 19 |
| univariate_90 | Moirai | 800 | 0 |
| univariate_90 | TimesFM | 800 | 0 |
| univariate_90 | XGBoost | 800 | 0 |
| univariate_90 | last_value | 800 | 0 |
| univariate_90 | weekly_seasonal_naive | 800 | 0 |
| univariate_180 | ARIMA | 800 | 2 |
| univariate_180 | Chronos-2 | 800 | 0 |
| univariate_180 | ETS | 800 | 1 |
| univariate_180 | Moirai | 800 | 0 |
| univariate_180 | TimesFM | 800 | 0 |
| univariate_180 | XGBoost | 800 | 0 |
| univariate_180 | last_value | 800 | 0 |
| univariate_180 | weekly_seasonal_naive | 800 | 0 |
| business_features_365 | XGBoost | 800 | 0 |

Total: 20,000 scheduled model/series/origin jobs; 42 failures, retained without fallbacks. Horizon rows do not inflate the failure denominator. Every saved final forecast was independently checked for complete scheduling, target/date alignment, clipping and identical overlapping prefixes; every score was recalculated and matched.

Saved in `results/metrics/stage7-final-57324046c857/`:

- Per-history, per-horizon mean-only and native-estimator accuracy.
- Across-series reliability (median/p90/p95, spread and worst cases) and shared-complete-series win rates.
- Paired 95% bootstrap intervals for all model pairs and MAE/RMSSE/WAPE: 2,000 whole-series draws, seed 42, preserving all origins and horizons. A negative A-minus-B difference favors A. Support is complete common series; intervals are descriptive, unadjusted for multiple comparisons, and shared store/product dependence limits population claims.
- `resource_summary.csv` and `resources.csv`: fitting/loading/first inference/warm timings and sampled RSS at the measured per-series, pooled or batched scope. Shared peak memory is not additive. Statistical workers are reused, so their RSS can retain allocations from earlier jobs. Times are host-specific; no profit or inventory savings are inferred.
- `segments.csv`, `origin_distributions.csv`, `case_scores.csv`, and `provenance.json`.

## Interpretation limits

Moirai's published family pretraining corpus includes M5; no clean M5-disjoint pretraining guarantee is established for these checkpoints. Final holdout discipline in this project does not undo possible external pretraining overlap. Chronos-2's median must not be described as a mean. Moirai's selected checkpoint is noncommercial. The diversity-balanced pilot excludes cold starts and does not represent every retail product.

No model choices were changed from final-test performance. Any new tuning, feature expansion, or model selection after this evaluation requires a fresh holdout or explicit exploratory labeling. This stage completes the planned shared-protocol runs; broader interpretation and business conclusions remain subsequent work.
