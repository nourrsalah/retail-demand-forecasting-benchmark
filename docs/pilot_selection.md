# Pilot selection (Stage 2)

Configuration: `configs/pilot.json`. Implementation: `src/data/pipeline.py`.

The sample contains 200 product–store series, selected with NumPy's seeded generator (seed 42). The manifest is committed as `results/metrics/pilot_manifest.csv`; raw data and processed daily data are ignored by Git.

## Historical information boundary

All selection statistics use d_1–d_730 (2011-01-29 through 2013-01-27). No later demand, price, or test outcomes affect selection. This cutoff must precede every eventual validation/test origin. Inspecting the full dataset is useful for integrity checks, but do not use later demand plots to alter the sample.

Series with no positive demand in that prefix are excluded because their demand characteristics cannot yet be measured. This excludes later-launch products and limits conclusions about new-product cold starts. Short-history experiments will simulate restricted model input on this established-product cohort; they are not a genuine new-product study.

## Characteristics and allocation

For each series, compute daily mean units, fraction of zero-demand days, and population standard deviation divided by mean (CV), including zeros in the prefix.

Exclusive demand labels, applied in order:

1. **Intermittent:** zero fraction >= 0.50.
2. **Volatile:** remaining series with CV >= 1.00.
3. **Stable:** remaining series with CV < 1.00 (a relative operational label, not a statistical guarantee).

Separately, assign low/medium/high volume from ranked mean-demand thirds within each store/category among eligible series. Equal means are broken by original source row order. These volume bands overlap the demand labels.

Within each store/category, create queues for each observed demand-type/volume-band combination. Shuffle queue members and stratum order with the fixed RNG. Cycle through strata to form a balanced group queue. Shuffle the store/category order, then draw round-robin until 200 series are selected. With 30 populated store/category groups, each gets six or seven series.

This is a diversity-oriented sample, not a population-proportional sample. Overall benchmark averages describe this pilot, not all Walmart demand. Rare characteristics may be overrepresented; always show segment counts. Exact reproducibility assumes the same raw-file hashes and pinned package versions.

## Data transformation and quality rules

Melt only selected series, retaining every observed sales day. Keep item, department, category, store, and state identifiers. Calendar joins use `d`; prices join on `(store_id, item_id, wm_yr_wk)`. Both joins enforce many-to-one cardinality. Require one row per series/date, complete dates, and nonnegative observed unit sales.

Missing event names mean no recorded event in that slot. Missing prices remain missing with a `price_missing` indicator; do not fill them with zero or use future prices. A zero sale does not establish whether an item was stocked, listed, or unavailable. No inventory or stockout observations are supplied here.

Outputs: `pilot_daily.csv.gz`, `pilot_series.csv`, and `series_profiles.csv` under `data/processed/`. Raw files are read only. Sampling membership and descriptive summaries are also saved under `results/metrics/`.
