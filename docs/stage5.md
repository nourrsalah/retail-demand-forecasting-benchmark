# Stage 5 — pooled XGBoost, univariate validation

This stage adds the locally trained machine-learning family under the locked primary experiment: 200 pilot series, 365 daily observations, the same three validation origins, horizons 7/14/28, and equal horizon weights for model selection. It does not run final tests or modify the Stage 4 selections.

## Prespecified design

`configs/stage5.json` fixes the search before validation scoring. One XGBoost model is fitted across all 200 pilot series per candidate and origin. This is pooled/global fitting, whereas ARIMA/ETS fit separately per series. The permitted information is sales history only; series IDs are used to align rows and outputs, not as predictors.

Eleven features: lagged sales at 1, 2, 7, 14, and 28 days; and trailing 7/28-day means, population standard deviations, and zero fractions. All features for a training target end on the preceding day. The first 28 targets of each allowed 365-day window are dropped; each series supplies 337 examples, for **67,400 training rows per fit**. Features and labels never borrow history before the window or after the forecast origin.

Fit a one-step squared-error regressor and forecast recursively for 28 days. Later forecast steps use earlier nonnegative-clipped predictions, never actual future sales. Preserve every raw prediction before clipping. No observed calendar, price, event, SNAP, store/category/state, or series-identity features enter this experiment. Those belong to the separately labeled business-feature experiment; history-length sensitivity remains later work.

Three candidates: depth 3 with 150 rounds, depth 3 with 300 rounds, and depth 6 with 200 rounds. All use learning rate 0.05, CPU histogram trees, full row/column sampling, L2 penalty 1, seed 42, and four training threads. Targets are not transformed or standardized, and training rows have equal weights. Consequently high-volume series can dominate the squared-error fitting objective; validation selection still follows macro RMSSE. There is no early stopping or post-result grid expansion.

Nine pooled fits are scheduled: three candidates × three origins. One configuration is selected across the entire pilot using the existing common-case and equal-horizon RMSSE selection helper, with MAE tie-breaking/fallback unchanged. These are a small fixed search and a recursive implementation, not exhaustive evidence about every possible XGBoost design.

## Safety and reproducibility

Workers receive only a series-by-day training matrix. Held-out actuals remain in the parent scorer. The runner first removes all dates after the final validation target (2016-01-31); it exposes no test execution option. Later validation origins can use already-observed earlier validation demand, as allowed by rolling backtesting.

Each pooled fit runs in a fresh worker process with four XGBoost threads, executed sequentially. The 30-minute timer covers feature preparation, fitting, first inference, and three warm inference repetitions. Exceptions/timeouts/invalid forecasts fail the pooled job and all affected series forecasts; no statistical model is substituted. Both pooled-job and affected-series failure counts are reported.

Timing logs separate feature preparation, fit, first inference, and three warm repeats. Sampled RSS is measured at 10ms intervals in the worker and includes process/library overhead; brief peaks can be missed. Import/startup and serialization overhead is included in whole-run wall time, not fit time. POSIX timer delivery may wait for native code to return. Do not divide pooled fit time and label it a measured per-series fit time. Compare the common CPU budget carefully: Stage 4 used four simultaneous single-thread workers; this stage uses one four-thread worker.

Run records contain software/hardware details, source/config/protocol/input hashes, exact candidates, origins, and seed. No model weights or fitted objects are serialized. Prior results are preserved in their own directories.

## Environment

XGBoost 3.4.1 requires the OpenMP native runtime on this Mac. The verified installation uses Homebrew libomp 23.1.2. See `docs/environment_setup.md` for setup instructions and the installation side effect recorded in this session.

## Run and inspect

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.evaluation.stage5 --smoke
python -m src.evaluation.stage5
python -m src.evaluation.report_stage5 results/metrics/RUN_ID
```

The smoke subset is the first frozen manifest ID per demand type and only the first validation origin; its rankings are diagnostic, not full-pilot model choices. Full run artifacts include candidate and selected forecasts, metrics, coverage, configuration selection, reliability, feature gain importance, warnings/failures, and pooled resource measurements. Gain importance describes fitted tree use, not causal drivers of sales.

Comparisons to Stage 4 must use matching histories, targets, manifest, and common scoring cases. They remain validation comparisons on data used to choose settings; final conclusions await the shared final-test stage. Bootstrap comparisons, business features, and 90/180-day sensitivity are outside this primary validation step.

Reference: [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html). The installed version is pinned in `requirements.txt` and recorded in each run.
