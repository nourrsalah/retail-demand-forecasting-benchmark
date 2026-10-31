# Retail Demand Forecasting Benchmark

**When do pretrained time-series models give a real advantage for retail demand forecasting, and when do classic local models remain the better choice?**

This project benchmarks **8 forecasting methods** on **200 Walmart (M5) product–store series**, from naive baselines and ARIMA/ETS to pooled XGBoost and three pretrained foundation models (Chronos-2, TimesFM 2.5, Moirai 1.1). All model choices were **frozen on validation data before the final test**, and every model was scored on accuracy, reliability, failures, runtime and memory.

## Key findings

- **Pretrained models gave real but modest gains.** TimesFM had the best scaled error (RMSSE), **1.5% lower than ARIMA**. Chronos-2 had the best absolute error, with **MAE 6.7% lower than ARIMA**, but it ranked behind ARIMA on RMSSE. No single model won on every metric.
- **The gains come at a cost.** TimesFM used **~10× ARIMA's memory** (1,530 MiB vs 152 MiB) and needed model downloads and a separate environment. ARIMA and ETS stayed close in accuracy and are cheap to run.
- **Short history:** pretrained models stayed strong with only 90 days of history. ARIMA and ETS failed to converge on 42 of 20,000 forecast jobs, all at 90 or 180 days of history.
- **Business features (calendar, events, SNAP, prices) did not clearly help XGBoost.** RMSSE improved slightly, but MAE and WAPE got worse.
- **Recommendation:** keep ARIMA/ETS as low-cost defaults. Consider TimesFM when a small gain in scaled error justifies the extra infrastructure, and Chronos-2 when absolute error matters most.

![Primary accuracy comparison](results/figures/stage7-final/accuracy_comparison.png)

## Results (365-day history, final test)

| Method | RMSSE ↓ | MAE ↓ | WAPE ↓ | Peak memory (MiB) |
| --- | ---: | ---: | ---: | ---: |
| **TimesFM 2.5** | **0.6914** | 1.0751 | 76.26% | 1,530 |
| ARIMA | 0.7019 | 1.1050 | 78.38% | 152 |
| ETS | 0.7029 | 1.1073 | 78.55% | 152 |
| XGBoost (pooled) | 0.7203 | 1.1019 | 78.17% | 205 |
| Moirai 1.1 small | 0.7245 | 1.1453 | 81.24% | 567 |
| **Chronos-2** | 0.7291 | **1.0313** | **73.15%** | 677 |
| Last value | 0.9097 | 1.3791 | 97.83% | 152 |
| Weekly seasonal naive | 0.9212 | 1.3184 | 93.52% | 152 |

Scores are averaged with equal weight over 7-, 14- and 28-day horizons and four final forecast origins (Jan–Apr 2016). Chronos-2's figures come from its native median forecast. All other models report mean forecasts, and a separate mean-only comparison is included in the evidence folder. Paired bootstrap 95% intervals show that TimesFM's RMSSE advantage over ARIMA is real: ARIMA − TimesFM = 0.0106 [0.0067, 0.0146]. Full interpretation: **[final comparison report](docs/final_comparison_report.md)**.

## How the evaluation works

- **Data:** 200 established product–store series from the [M5 competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy), sampled to balance stores, states, categories and demand types (intermittent, stable, volatile). See [pilot selection](docs/pilot_selection.md).
- **No test leakage:** models were tuned on three earlier validation origins. All choices were then frozen in `configs/stage7_freeze.json`, with source and config hashes, before the four test origins were run.
- **Scale:** 20,000 scheduled forecast jobs across 90/180/365-day histories. **99.79% succeeded**, and the 42 failures were kept in the results instead of being replaced with fallback forecasts.
- **Metrics:** RMSSE (the primary metric), MAE and WAPE; per-series reliability percentiles; win rates; runtime and peak memory.
- **Uncertainty:** paired bootstrap confidence intervals that resample whole series.
- **Pretraining audit:** Moirai's pretraining corpus includes M5, and its checkpoint carries a non-commercial license. These limits are documented in the [pretraining audit](docs/pretraining_audit.md).

## Limitations

- The sample is balanced for diversity, so it is not representative of the full M5 population. Cold-start (new) products are excluded.
- Four test origins cannot cover every retail regime, such as holiday peaks.
- RMSSE here is bottom-level and unweighted, which differs from the official hierarchical M5 WRMSSE.
- Forecast accuracy is not the same as business value. Inventory savings would require a separate decision model that includes lead times, stockout costs and service levels.

## Repository structure

```
configs/       Frozen evaluation, model and experiment settings
docs/          Protocol, per-stage design and results, final report, pretraining audit
notebooks/     Data understanding and pilot dataset construction
src/data/      Loading, auditing and pilot selection pipeline
src/models/    Statistical, XGBoost and foundation-model adapters
src/evaluation/ Windows, metrics, stage runners and report generation
tests/         Unit tests for the pipeline and each stage
results/       Metrics, forecasts and figures (final results in results/metrics/stage7-final-*)
```

## Reproduce

Requires Python 3.12. On macOS, XGBoost also needs `libomp` (see [environment setup](docs/environment_setup.md)). Download the three original M5 CSVs into `data/raw/` first.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.data.pipeline
python -m unittest discover -s tests -v

# Statistical baselines and XGBoost (validation)
python -m src.evaluation.stage4
python -m src.evaluation.stage5

# Foundation models run in an isolated environment
python -m venv .venv-stage6
.venv-stage6/bin/python -m pip install -r requirements/stage6.lock.txt
.venv-stage6/bin/python -m src.utils.prepare_stage6
python -m src.evaluation.stage6

# Final test and reports
python -m src.evaluation.benchmark test
python -m src.evaluation.report_benchmark
python -m analysis.stage7_history
```

The final test periods have already been evaluated. Any new experiment on them should be treated as a follow-up robustness study and must not be used to retune models. The full workflow and change-control rules are in the [evaluation protocol](docs/evaluation_protocol.md) and the [Stage 7 design](docs/stage7_design.md).

## Project stages

1. Data understanding → 2. Pilot dataset → 3. Evaluation protocol → 4. Baselines, ARIMA, ETS → 5. XGBoost → 6. Chronos-2, TimesFM, Moirai → 7. Frozen final test and business interpretation. Each stage has its own design and results document in `docs/`.
