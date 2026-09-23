# Stage 4 validation results

Run: `20260923T142001461574Z-stage4-validation`. **Validation only; final test has not been evaluated.**

Completed 4,800 scheduled tasks across 200 series, three validation origins, and eight prespecified candidates. Wall time: 24.0 seconds with 4 CPU workers; this includes data preparation and artifact writing. Failed tasks: 3. Predictions were not substituted for failures.

## Selected specifications

- ARIMA: `arima_011`.
- ETS: `ets_ann`.
- Reference rules: last value and weekly seasonal naive.

Selection uses equal 1/3 horizon weights. Candidate scores below use common successful support **within each family**, so selection scores should not be compared across families without checking support.

| model | candidate | selection_score | common_scored_cases | failed_cases |
| --- | --- | --- | --- | --- |
| ARIMA | arima_100 | 0.7016 | 597 | 0 |
| ARIMA | arima_011 | 0.6601 | 597 | 1 |
| ARIMA | sarima_100_100_7 | 0.6945 | 597 | 2 |
| ETS | ets_ann | 0.6608 | 600 | 0 |
| ETS | ets_aadn | 0.6618 | 600 | 0 |
| ETS | ets_ana_7 | 0.6628 | 600 | 0 |
| last_value | last_value | 0.8730 | 600 | 0 |
| weekly_seasonal_naive | weekly_seasonal_naive | 0.8872 | 600 | 0 |

## Matched descriptive comparison

This table averages each horizon's aggregate metric with equal weight. WAPE is first pooled within each horizon, never averaged across series. The underlying horizon table is `selected_comparison_common_cases.csv`. All selected methods share the same eligible series/origin cases at each horizon (599–599 cases, out of 600). RMSSE eligibility also defines this table's MAE/WAPE support; full valid-MAE coverage is in `accuracy_by_horizon.csv`.

| model | candidate | MAE | RMSSE | WAPE |
| --- | --- | --- | --- | --- |
| ARIMA | arima_011 | 1.0655 | 0.6609 | 0.8286 |
| ETS | ets_ann | 1.0661 | 0.6612 | 0.8291 |
| last_value | last_value | 1.4200 | 0.8737 | 1.1046 |
| weekly_seasonal_naive | weekly_seasonal_naive | 1.3109 | 0.8876 | 1.0195 |

These scores are used for tuning and do not constitute final-test evidence. They cannot establish conclusions about pretrained versus locally trained models, since pretrained models are not included yet.

## Reliability, failures, and resources

`reliability.csv` reports median/p90/p95/IQR/standard deviation/maximum across series with all three origins valid. `win_rates.csv` uses common complete series across selected methods and shares tied wins. These are validation-only descriptions. `origin_distributions.csv` exposes period-level variability.

Failures:

| candidate | series_id | forecast_origin | error |
| --- | --- | --- | --- |
| arima_011 | HOBBIES_1_012_TX_1_evaluation | 2016-01-03 | NonConvergenceError: Optimizer did not converge within the prespecified budget |
| sarima_100_100_7 | HOBBIES_2_008_CA_3_evaluation | 2015-11-08 | NonConvergenceError: Optimizer did not converge within the prespecified budget |
| sarima_100_100_7 | HOUSEHOLD_2_375_WI_3_evaluation | 2016-01-03 | NonConvergenceError: Optimizer did not converge within the prespecified budget |

`resources.csv` records per-task fit, first inference, three-repeat warmed inference, warnings, and sampled process memory. `resource_summary.csv` summarizes worker measurements; summed worker seconds are not elapsed wall time. Hardware and measurement limitations are recorded in `run.json` and `docs/stage4.md`. Exact source/configuration hashes and software versions accompany the run.

## Files and next boundary

- Metrics and configuration selection: `results/metrics/20260923T142001461574Z-stage4-validation/`.
- Candidate and selected forecasts: `results/forecasts/20260923T142001461574Z-stage4-validation/`.
- Figures: `results/figures/20260923T142001461574Z-stage4-validation/`.

The 365-day Stage 4 validation experiment is complete. Keep the statistical choices recorded while proceeding to other model families. Do not run the four final test origins until all relevant model and experiment choices are frozen. The 90/180-day sensitivity conditions remain later work.
