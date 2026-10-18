# Stage 7 execution choices — frozen before test

This implements the existing protocol v1.0 without changing the pilot, origins, horizons, information tracks, metrics or equal horizon weighting.

## Validation and freeze

Reuse the audited 365-day univariate validation predictions from Stages 4–6. Refit the same eight statistical/reference candidates and three XGBoost candidates separately at 90 and 180 days, with identical optimizer limits, timeouts and search budgets. Use one prespecified frozen checkpoint for each foundation family at every history length. Select candidates independently per family/history/scenario using the existing common-support RMSSE objective and MAE tie break. Never choose settings from final-test results.

Foundation adapters accept exactly 90, 180 or 365 observed targets. Chronos receives the requested context length; Moirai's context and dataset start are adjusted to that exact history. TimesFM allocates the smallest aligned context: 96/192/384 slots for 90/180/365 observations, with masked padding only. An initial validation attempt using 384 slots for 90 observations produced nonfinite outputs in the installed runtime; avoiding fully masked leading patches fixes that input-compatibility problem without adding history or changing weights. The failed attempt is retained and was not used for selection. No observations outside the permitted window enter inference. Checkpoints, estimator extraction, batch size, seed, precision and thread settings are unchanged. Short histories are limited-history experiments, not real cold starts.

`configs/stage7_freeze.json` records all selected candidates, experiment choices, checkpoint revisions, validation evidence and source/data hashes before any final-test forecast. Test execution rejects changed source, configs, data or validation evidence. Completed test conditions are preserved; they are never used to change selections. Source model fitting differs by family: per-series ARIMA/ETS, pooled XGBoost, frozen foundation checkpoints.

## Separate business-feature condition

The secondary condition is **365-day XGBoost with business features**, compared with the independently selected sales-only XGBoost on identical targets. This is a within-model information experiment, not a controlled foundation-versus-local comparison. The three XGBoost candidates and fitting budget are unchanged.

Features add day-of-week/year sine/cosine, an event indicator, scheduled SNAP for the series' state, previous available sell price, and a missing-price indicator. Calendar events and SNAP are explicitly assumed scheduled and known at the origin. Training price features use only earlier prices within the permitted history; future forecasts carry forward the final price observed by the origin. Realized future prices and future sales are never passed to the model. Zero is used for unavailable historical prices alongside an explicit missing indicator. State routes the SNAP schedule, not a learned store/product/category embedding. Calendar events are not asserted to be actual promotions; M5 lacks inventory observations.

Other model covariate adapters are **not implemented in this experiment**; this is not a statement that those model families cannot use covariates. No shorter-history business-feature condition or accuracy-driven expansion of feature sets will be added after this test. Such extensions require fresh evaluation data or an explicitly exploratory label. The optional seasonality flag is not used; original frozen demand/volume labels remain.

## Final evaluation

After the freeze, evaluate each selected univariate family at all three history lengths on the four original final-test origins. Refit local models separately at each origin using that origin's permitted history; earlier test actuals become history only after their forecast window closes. Run the separate 365-day business-feature XGBoost condition using its frozen validation selection. Generate 28 predictions once, then score prefixes 7/14/28 identically.

Report mean-only and explicitly labeled native-estimator comparisons, common-support accuracy with full failure counts, horizon/origin results, series consistency, fixed historical segments, and paired series bootstrap intervals (2,000 draws, seed 42, 95% percentile). Preserve all origins/horizons within bootstrap draws. Intervals are descriptive, unadjusted for multiple comparisons and do not remove dependence between related products/stores. RMSSE is compared within each history condition; MAE/WAPE quantify changes across history lengths.

Retain original and newly measured resource logs at their actual per-series, pooled-origin or batched-origin scope. Three warm repeats remain enabled. Memory is sampled, includes runtime overhead and is not summed across rows. The statistical process pool can reuse workers, as in Stage 4; sampled RSS may include allocations from earlier tasks and is not a fresh-process measurement. Zero-shot model loading and checkpoint downloads are distinct from inference. Business-feature XGBoost fit timing includes construction of its lag/feature matrix; calendar/price input assembly is included only in total condition wall time, not a separately measured feature timer.

Moirai's disclosed family pretraining data includes M5. No clean M5-disjoint pretraining claim is made for any selected checkpoint. Chronos-2 is a median forecast and excluded from mean-only comparisons; Moirai is a 200-sample mean with a noncommercial checkpoint license. These limits also apply to final test results.

Commands, from project root:

```sh
.venv/bin/python -m src.evaluation.stage7
.venv/bin/python -m src.evaluation.benchmark validate
.venv/bin/python -m src.evaluation.benchmark freeze
.venv/bin/python -m src.evaluation.benchmark test
.venv/bin/python -m src.evaluation.report_benchmark
```
