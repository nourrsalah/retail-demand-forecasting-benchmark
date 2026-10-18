# Retail Demand Forecasting Benchmark

**Research question:** When do pretrained time-series models provide a real advantage for business demand forecasting, and when do locally trained models remain more reliable?

Planned comparisons: ARIMA, ETS, XGBoost, Chronos-2, TimesFM, and Moirai, with simple naive references. The focus is accuracy, reliability across demand conditions, and operational practicality—not one universal winner.

## Current scope

Stages 1 and 2 are complete: data understanding and a reproducible 200-series pilot. Stage 3’s core design is locked in [the evaluation protocol](docs/evaluation_protocol.md) and `configs/evaluation.json`: 365-day primary history, later 90/180/365 sensitivity, and separate univariate/business-feature experiments.

Stage 4 implements last-value and weekly seasonal-naive references, ARIMA, and ETS. See [Stage 4](docs/stage4.md) for prespecified candidates, validation-only execution, resource measurements, and limitations. Stages 4–6 use validation only; final-test execution is handled separately by Stage 7 after choices are frozen. The completed 200-series validation run is summarized in [Stage 4 results](docs/stage4_results.md).

Stage 5 adds pooled XGBoost using sales-history features and recursive forecasts under the same univariate validation protocol. See [Stage 5 design](docs/stage5.md) and [Stage 5 results](docs/stage5_results.md). The business-feature and shorter-history validation experiments are implemented in Stage 7.

Stage 6 adds frozen Chronos-2, TimesFM 2.5, and Moirai 1.1 small adapters in an isolated environment. See [Stage 6 design](docs/stage6.md) and the [pretraining audit](docs/pretraining_audit.md). The completed 200-series validation run is summarized in [Stage 6 results](docs/stage6_results.md): all 1,800 model/series/origin forecasts succeeded. Checkpoints are pinned and ignored by Git; Stage 6 itself uses validation only.

Stage 7 completes independent validation selection for 90/180-day histories and a separate 365-day business-feature XGBoost condition. All choices are frozen in `configs/stage7_freeze.json` before the four final test origins. See [Stage 7 execution design](docs/stage7_design.md). Final evaluation is complete: 20,000 scheduled forecast jobs, with 42 failures retained without fallbacks. See the [final comparison report](docs/stage7_final_results.md).

## Setup and reproduce

Use Python 3.12. XGBoost also requires libomp on macOS; see [environment setup](docs/environment_setup.md). From the project root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.data.pipeline
python -m unittest discover -s tests -v
# Stage 4: smoke check, then the full validation pilot
python -m src.evaluation.stage4 --smoke
python -m src.evaluation.stage4
# Stage 5: pooled XGBoost validation
python -m src.evaluation.stage5 --smoke
python -m src.evaluation.stage5
```

Place the three original M5 CSVs in `data/raw/` before running. In VS Code, select `.venv/bin/python` as the notebook kernel and run notebooks from the `notebooks/` working directory. The notebooks contain saved outputs; the pipeline rebuilds their underlying tables/figures, and rerunning the notebooks refreshes displayed results.

The local environment used for this run is Python 3.12 with access to bundled pandas/NumPy packages; the requirements above allow a standalone environment to be built without that bundle. Run provenance includes actual package versions and source-file SHA-256 hashes.

### Stage 7 execution and reporting

The existing checkout is frozen. Resume its scheduled final conditions and regenerate reports with:

```bash
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python -m src.evaluation.benchmark test
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python -m src.evaluation.report_benchmark
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python -m analysis.stage7_history
```

Completed test conditions are reused, not rerun or retuned. The last command creates the identical-target history comparison and updates the report. Its analysis source was also hashed before testing. The source/data/configuration freeze is verified before test execution.

For a fresh benchmark before any test access, the execution order is validation, model-choice freeze, supplementary-analysis freeze (`python -m analysis.stage7_history --freeze`), then test and reporting. The current validation selections and test results must not be used to retune models on the same holdout.

## Where to look

- `notebooks/01_data_understanding.ipynb`: original user cells plus missingness, hierarchy, event/SNAP, price coverage, and demand plots.
- `notebooks/02_pilot_dataset.ipynb`: reproducibility checks, sample composition, and joined daily data.
- `src/data/pipeline.py`: reusable loading, audit, profiling, selection, transformation, and figure generation.
- `configs/pilot.json`: sample size, seed, historical cutoff, and demand thresholds.
- `docs/pilot_selection.md`: exact selection procedure and limitations.
- `docs/evaluation_protocol.md`: reviewed v1.0 evaluation rules and change-control policy.
- `data/README.md`: source schema and missing-data interpretation.
- `data/processed/`: ignored derived profiles, selected series, and compressed long-form pilot.
- `results/metrics/`: audit, pilot membership manifest, hierarchy/event tables, and run provenance; includes data summaries plus separate Stage 4 validation run directories.
- `results/figures/`: representative historical demand and sample composition.
- `src/models/statistical.py`: statistical and reference adapters.
- `src/models/foundation.py`: frozen pretrained adapters; `src/evaluation/stage6.py` runs their validation.
- `configs/stage6.json`: frozen checkpoint choices, estimators, CPU and batch settings.
- `src/models/xgboost_model.py`: lag/rolling features, pooled fitting, and recursive forecasts.
- `src/evaluation/`: validation windows, metrics, candidate selection, and Stage 4 runner.
- `results/forecasts/`: compressed candidate and selected-model validation forecasts.
- `configs/stage4.json`: prespecified statistical search and resource settings.
- `configs/stage5.json`: prespecified XGBoost search, features, and resource settings.

## Pilot design

Sample 200 established product/store series across all stores, states, and categories. Selection statistics use only d_1–d_730, ending 2013-01-27. Round-robin sampling balances store/category groups and observed demand/volume strata, with seed 42. Melt the selected data and attach calendar/weekly prices using validated joins. Missing prices remain missing.

The sample intentionally emphasizes diversity, so its average results are not population estimates. Products inactive throughout the sampling prefix are excluded; simulated limited-history tests cannot establish real new-product performance. Raw data are never overwritten, and both raw and processed data are ignored by Git.

## Roadmap

1. Data understanding.
2. Create pilot dataset.
3. Shared evaluation rules (core design and Stage 7 execution choices frozen).
4. Simple baselines, ARIMA, and ETS.
5. XGBoost.
6. Chronos-2, TimesFM, and Moirai.
7. Run the shared protocol.
8. Consolidate results.
9. Analyze when pretrained versus local models work better.
10. Business practicality and final conclusions.

Keep notebooks exploratory, model code reusable, and test outcomes out of sampling, tuning, and protocol decisions. GitHub publication, broader result interpretation, and final business conclusions remain later work.
