# Retail Demand Forecasting Benchmark

**Research question:** When do pretrained time-series models provide a real advantage for business demand forecasting, and when do locally trained models remain more reliable?

Planned comparisons: ARIMA, ETS, XGBoost, Chronos-2, TimesFM, and Moirai, with simple naive references. The focus is accuracy, reliability across demand conditions, and operational practicality—not one universal winner.

## Current scope

Stages 1 and 2: data understanding and a reproducible 200-series pilot. **No forecasting models are implemented.** Stage 3’s core design is locked in [the evaluation protocol](docs/evaluation_protocol.md) and `configs/evaluation.json`: 365-day primary history, later 90/180/365 sensitivity, and strictly separate univariate/business-feature experiments. Model implementation has not started.

## Setup and reproduce

Use Python 3.12. From the project root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.data.pipeline
python -m unittest discover -s tests -v
```

Place the three original M5 CSVs in `data/raw/` before running. In VS Code, select `.venv/bin/python` as the notebook kernel and run notebooks from the `notebooks/` working directory. The notebooks contain saved outputs; the pipeline rebuilds their underlying tables/figures, and rerunning the notebooks refreshes displayed results.

The local environment used for this run is Python 3.12 with access to bundled pandas/NumPy packages; the requirements above allow a standalone environment to be built without that bundle. Run provenance includes actual package versions and source-file SHA-256 hashes.

## Where to look

- `notebooks/01_data_understanding.ipynb`: original user cells plus missingness, hierarchy, event/SNAP, price coverage, and demand plots.
- `notebooks/02_pilot_dataset.ipynb`: reproducibility checks, sample composition, and joined daily data.
- `src/data/pipeline.py`: reusable loading, audit, profiling, selection, transformation, and figure generation.
- `configs/pilot.json`: sample size, seed, historical cutoff, and demand thresholds.
- `docs/pilot_selection.md`: exact selection procedure and limitations.
- `docs/evaluation_protocol.md`: reviewed v1.0 evaluation rules and change-control policy.
- `data/README.md`: source schema and missing-data interpretation.
- `data/processed/`: ignored derived profiles, selected series, and compressed long-form pilot.
- `results/metrics/`: audit, pilot membership manifest, hierarchy/event tables, and run provenance; these are data summaries, not forecasting scores.
- `results/figures/`: representative historical demand and sample composition.
- `src/models/`, `src/evaluation/`, `src/utils/`, and `results/forecasts/`: reserved for later stages.

## Pilot design

Sample 200 established product/store series across all stores, states, and categories. Selection statistics use only d_1–d_730, ending 2013-01-27. Round-robin sampling balances store/category groups and observed demand/volume strata, with seed 42. Melt the selected data and attach calendar/weekly prices using validated joins. Missing prices remain missing.

The sample intentionally emphasizes diversity, so its average results are not population estimates. Products inactive throughout the sampling prefix are excluded; simulated limited-history tests cannot establish real new-product performance. Raw data are never overwritten, and both raw and processed data are ignored by Git.

## Roadmap

1. Data understanding.
2. Create pilot dataset.
3. Shared evaluation rules (core design locked; model execution details pending).
4. Simple baselines, ARIMA, and ETS.
5. XGBoost.
6. Chronos-2, TimesFM, and Moirai.
7. Run the shared protocol.
8. Consolidate results.
9. Analyze when pretrained versus local models work better.
10. Business practicality and final conclusions.

Keep notebooks exploratory, model code reusable, and test outcomes out of sampling, tuning, and protocol decisions. GitHub publication and all model work are outside the current stage.
