# Data

Place these M5 source files in `raw/` (never modify them):

| File | Row meaning | Key / main fields |
|---|---|---|
| calendar.csv | One calendar day | `d`, `date`, `wm_yr_wk`, calendar fields, two event slots, state SNAP indicators |
| sales_train_evaluation.csv | One item/store series | `id`, item/department/category/store/state, daily units in `d_1`…`d_1941` |
| sell_prices.csv | One item/store/week price | `item_id`, `store_id`, `wm_yr_wk`, `sell_price` |

`wm_yr_wk` is the shared retail-week identifier. Join sales to calendar via `d`, then prices via store/item/week. The source calendar extends beyond the labeled sales period.

`processed/` contains derived pilot files, never source replacements. Missing prices stay missing, with an indicator; absent event slots are expected. Zero units do not distinguish no demand from lack of availability. See `docs/pilot_selection.md` for eligibility and sampling rules. All raw/processed contents are ignored by Git.
