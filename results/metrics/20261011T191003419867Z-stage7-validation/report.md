# Stage 7 — unified validation evaluation

All eight methods are evaluated under protocol v1.0: 200 pilot series, the same three validation origins, 365 days of sales history, and 7/14/28-day horizons with equal 1/3 weights. **Final test periods remain untouched.**

Run: `20261011T191003419867Z-stage7-validation`. This execution reuses completed model predictions after checking their provenance, complete schedule, daily targets, dates, nonnegative clipping and identical horizon prefixes. Every score was recalculated from the pilot and checked against its source. Models were not retrained, retuned or rerun; original measured resource logs are retained.

## Shared accuracy

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

Lower errors are better. There are 599 common successful series/origin cases per horizon out of 600. The table uses this common support for every metric; WAPE pools absolute errors and sales within a horizon, then averages the three horizon values equally. Chronos-2 supplies a median; the others use the documented point forecasts. This is the **native-estimator** comparison; `summary_mean_only.csv` excludes Chronos-2. These are selected-model validation scores, not final test performance.

## Failures and uncertainty

| model | scheduled_jobs | failed_jobs |
| --- | --- | --- |
| ARIMA | 600 | 1 |
| Chronos-2 | 600 | 0 |
| ETS | 600 | 0 |
| Moirai | 600 | 0 |
| TimesFM | 600 | 0 |
| XGBoost | 600 | 0 |
| last_value | 600 | 0 |
| weekly_seasonal_naive | 600 | 0 |

The single earlier ARIMA failure is preserved without fallback. Failure counts use 4,800 scheduled model/series/origin jobs, not duplicated horizon rows. Reliability summaries require all three origins per series and include median, p90, p95, spread, maximum and exclusions. Win rates use common complete series and split ties.

`paired_intervals_*.csv` reports all prespecified model pairs for MAE, RMSSE and WAPE using 2,000 paired whole-series bootstrap draws (seed 42), retaining all origins and horizons. Each comparison track uses its complete common series, so the intervals can have stricter support than the accuracy table. Differences are model A minus B; negative favors A. These are descriptive 95% percentile intervals, with no correction for multiple comparisons. Shared products/stores introduce dependence, and validation selection introduces optimism; these are not population-wide guarantees.

## Reproducibility and limitations

- `run.json`: source artifact hashes, evaluation settings, and successful audit checks.
- `case_scores.csv`, `metrics.csv`: newly calculated scores for all scheduled cases.
- `source_resources.csv`: original timings and sampled memory, labeled by per-series, pooled, or batched scope. No fresh runtime benchmark was performed; do not sum shared peak memory or interpret worker-time sums as wall time.
- `segments_native_estimators.csv`: fixed historical demand/volume groups, states, stores and categories, with support counts.
- Saved combined predictions: `results/forecasts/20261011T191003419867Z-stage7-validation/forecasts.csv.gz`.

Moirai's disclosed family pretraining corpus includes M5; disjoint pretraining is not certified for these checkpoints. See `docs/pretraining_audit.md`. Chronos-2's median and Moirai's noncommercial checkpoint restriction remain explicitly documented. This pilot excludes cold starts.

This completes the unified **365-day univariate validation evaluation**. Shorter-history selection, the separate business-feature experiment, freezing all remaining choices, and final-test evaluation remain pending under the locked protocol. This report does not authorize or execute test access.
