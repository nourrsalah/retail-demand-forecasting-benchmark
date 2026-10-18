# Stage 5 validation results — XGBoost

Run: `20260923T161927565554Z-stage5-validation`. **Validation only. No final-test forecasts or scores.**

Completed 9 pooled jobs (3 candidates × 3 origins) on 200 series. Each fit used 67,400 examples, drawn entirely from its 365-day history window. Features are past sales only. There were 0 failed pooled jobs, affecting 0 scheduled series cases. Full-run wall time: 15.1 seconds; pooled fit timing is logged separately.

## Configuration selection

Selected: **`xgb_depth3_rounds300`**. The fixed three-candidate search used equal 1/3 weights for 7/14/28-day macro RMSSE, with MAE tie-breaking. One fitted model shares information across all pilot series; identifiers are not predictors.

| candidate | selection_score | mae_tie_break | common_scored_cases | failed_cases |
| --- | --- | --- | --- | --- |
| xgb_depth3_rounds150 | 0.6848 | 1.0812 | 600 | 0 |
| xgb_depth3_rounds300 | 0.6822 | 1.0806 | 600 | 0 |
| xgb_depth6_rounds200 | 0.6855 | 1.0968 | 600 | 0 |

## Comparison with Stage 4

The previous Stage 4 choices are retained. All five methods are compared on the same 599–599 eligible series/origin cases per horizon, out of 600. Each number below equally averages the three horizon-level aggregates. WAPE is pooled within each horizon before that averaging. The table uses common RMSSE-eligible support for all three metrics; per-model full-coverage metrics are also saved.

| model | candidate | MAE | RMSSE | WAPE |
| --- | --- | --- | --- | --- |
| ARIMA | arima_011 | 1.0655 | 0.6609 | 0.8286 |
| ETS | ets_ann | 1.0661 | 0.6612 | 0.8291 |
| XGBoost | xgb_depth3_rounds300 | 1.0820 | 0.6826 | 0.8413 |
| last_value | last_value | 1.4200 | 0.8737 | 1.1046 |
| weekly_seasonal_naive | weekly_seasonal_naive | 1.3109 | 0.8876 | 1.0195 |

These are validation scores used to choose settings, not final-test estimates or evidence about pretrained models. The report does not declare a universal model-family winner. XGBoost's pooled learning differs from the per-series statistical fits; its raw-unit squared-error objective can emphasize larger-demand examples. No calendar or price information was supplied.

## Saved evidence

- `stage4_stage5_comparison.csv`: matched horizon-level scores.
- `comparison_by_segment.csv`: fixed demand/volume groups and store/category comparisons with support counts.
- `combined_reliability.csv`: median/p90/p95 and other per-series summaries; complete origins required per method.
- `combined_win_rates.csv`: common complete series across all five methods; tied wins are shared.
- `resources.csv` and `selected_resource_comparison.csv`: pooled versus per-series measurements explicitly labeled. Worker memory includes runtime overhead and is sampled; summed CPU-worker time is not wall time.
- `feature_gain.csv`: training-only tree gain importance, not causal effects.
- `selected_models.json`: selected configuration and hashes.
- `run.json`: immutable-at-run configuration, data/source hashes, software/hardware details, and history boundaries.

Forecasts: `results/forecasts/20260923T161927565554Z-stage5-validation/`. Chart: `results/figures/20260923T161927565554Z-stage5-validation/validation_comparison.png`.

Stage 4 source: `20260923T142001461574Z-stage4-validation`. Its raw pilot, manifest, horizons, origins, and history condition were checked for compatibility before comparison. Neither its selections nor any locked protocol decision was changed.

## Next boundary

The primary Stage 5 implementation and validation are complete. Business-feature experiments and 90/180-day sensitivity remain separate later work. Stage 6 adds pretrained models. Final-test execution waits until all model-family and experiment choices are frozen.
