# Stage 4 — statistical models and reference baselines

This stage implements and validates the 365-day **univariate** condition only. Final-test forecasting is deliberately absent from the runner. History sensitivity and business features remain later experiments under the locked protocol.

## Prespecified implementation choices

Configuration: `configs/stage4.json`, saved with its SHA-256 in each run before fitting starts.

| Family | Candidates | Estimation |
|---|---|---|
| Reference rules | Last observed value; repeat the final seven observed days | No fitted parameters |
| ARIMA | (1,0,0) with constant; (0,1,1) without drift; seasonal (1,0,0)(1,0,0)[7] with constant | State-space maximum likelihood; stationarity and invertibility enforced |
| ETS | ETS(A,N,N); ETS(A,Ad,N); ETS(A,N,A)[7] | Additive-error state-space exponential smoothing, estimated initialization |

The three candidates per fitted family have a shared 300-iteration optimizer cap and 120-second fit/forecast timer per series/origin/candidate. Additive ETS accommodates zero observations; multiplicative specifications are not in this fixed search. Weekly seasonality is a prespecified candidate, not selected by examining final test demand. Search scope is deliberately small; conclusions are about these configured implementations, not exhaustive optimization of each family.

Candidates are fitted separately to every pilot series at each validation origin. Select **one configuration per family across the pilot** rather than choosing a different configuration from each individual series' validation outcomes. Select by the equal-weighted mean of horizon-specific macro RMSSE (7/14/28 days); macro MAE breaks ties, with the same horizon weights. A remaining exact tie is broken lexically by candidate ID for reproducibility. If all training scales are undefined, use MAE as specified by the protocol.

For selection, restrict to common successful series/origin cases across candidates and all horizons within a family. Report this support, excluded cases, and each candidate's failures. Undefined RMSSE scales are excluded from RMSSE selection; no horizon's weight is redistributed. This avoids giving a candidate easier scoring cases merely because it failed elsewhere. No selection is made when the comparison has no valid support. Results on successful subsets alone do not establish general superiority.

An optimizer non-convergence, exception, timeout, or invalid prediction is a failed run, with no model fallback. Keep failed forecast rows with missing predictions and explicit status. Preserve raw forecasts and apply nonnegative clipping consistently before scoring; count negative raw values. Point forecasts use the adapters' conditional mean forecasts. No model receives calendar, price, event, or target values after the origin.

## Execute

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.evaluation.stage4 --smoke
python -m src.evaluation.stage4
# Replace RUN_ID with the completed full validation run directory name.
python -m src.evaluation.report_stage4 results/metrics/RUN_ID
```

The smoke run selects the first manifest ID in each frozen demand group and only the first validation origin. It checks execution; it does not select final model settings. The full validation run uses all 200 series and the three locked validation origins: 4,800 scheduled candidate/series/origin jobs. No test option is provided. A 28-day forecast is generated once and prefixes are scored at 7/14/28 days.

## Artifacts and interpretation

Every invocation creates a separate run directory under `results/metrics/` and `results/forecasts/`, preserving previous runs. Outputs include candidate forecasts, standard forecast/metric identifiers, per-case scores, pooled WAPE, horizon summaries, selection scores and selected settings, cross-series median/p90/p95 errors, origin distributions, win rates, failure/warning logs, and resource logs. Tuning artifacts include `candidate` in their keys because multiple specifications share one model-family label; final selected forecasts use the protocol's model-level key.

Validation scores are used for selection, so they are **not unbiased final benchmark results**. The later final-test evaluation must wait until all model-family choices and experiment configurations are finalized. Other families have not been implemented. Any later change to these statistical search settings requires a new run record and rationale, not overwriting this experiment.

Four worker processes each use one BLAS thread. Timing separates fit, first inference, and three warm inference repetitions. Total wall time includes loading, pool startup, and saving; summing worker timings is not wall time. CPU-only hardware, Python/package versions, configuration/protocol/source hashes, Git revision, and pilot hash are recorded. No serialized fitted models or checkpoints are saved.

Memory is sampled worker RSS at 10ms intervals and includes runtime overhead; brief peaks can be missed. It is not aggregate host memory or GPU memory. The timeout uses a POSIX interval timer; delivery may wait for native code to return. These measurement limits must accompany later resource comparisons. Package installation/download time is not retrospectively estimated. Stage 4 includes descriptive reliability tables; paired-bootstrap final comparisons belong to consolidation after all model families are available.

## Implementation references

- [statsmodels ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
- [statsmodels ETS examples](https://www.statsmodels.org/stable/examples/notebooks/generated/ets.html)

The exact installed statsmodels version is pinned in `requirements.txt` and recorded in every run.
