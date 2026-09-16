"""Stage 1/2 data audit and reproducible pilot. Run from project root as a module."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IDS = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']


def load_data(root=ROOT):
    raw = root / 'data/raw'
    calendar = pd.read_csv(raw / 'calendar.csv', parse_dates=['date'])
    sales = pd.read_csv(raw / 'sales_train_evaluation.csv')
    prices = pd.read_csv(raw / 'sell_prices.csv')
    return calendar, sales, prices


def day_columns(sales):
    return sorted([c for c in sales if c.startswith('d_')], key=lambda c: int(c[2:]))


def profile_series(sales, config):
    """All selection features use a fixed historical prefix, never future demand."""
    days = day_columns(sales)[:config['profile_days']]
    if len(days) != config['profile_days']:
        raise ValueError('Insufficient history for sampling profile')
    y = sales[days]
    profile = sales[IDS].copy()
    profile['mean_units'] = y.mean(axis=1)
    profile['zero_fraction'] = y.eq(0).mean(axis=1)
    profile['cv'] = y.std(axis=1, ddof=0).div(profile['mean_units'].replace(0, np.nan))
    profile['eligible'] = profile['mean_units'].gt(0)
    profile['demand_type'] = np.select(
        [~profile['eligible'], profile['zero_fraction'].ge(config['intermittent_zero_fraction']),
         profile['cv'].ge(config['volatile_cv'])],
        ['inactive_in_profile', 'intermittent', 'volatile'], default='stable')
    # Ranking breaks equal-volume ties consistently by input series order.
    ranks = profile.loc[profile.eligible].groupby(['store_id', 'cat_id'])['mean_units'].rank(method='first', pct=True)
    profile['volume_band'] = 'inactive'
    profile.loc[ranks.index, 'volume_band'] = pd.cut(ranks, [0, 1/3, 2/3, 1], labels=['low', 'medium', 'high']).astype(str)
    return profile


def select_pilot(profile, config):
    eligible = profile.loc[profile.eligible].sort_values('id')
    if len(eligible) < config['n_series']:
        raise ValueError('Not enough eligible series')
    rng = np.random.default_rng(config['seed'])
    # First balance store/category groups; cycle through their available demand/volume strata.
    queues = {}
    for key, group in eligible.groupby(['store_id', 'cat_id'], sort=True):
        strata = [list(rng.permutation(g.index)) for _, g in group.groupby(['demand_type', 'volume_band'], sort=True)]
        strata = [strata[i] for i in rng.permutation(len(strata))]
        queue = []
        while any(strata):
            for bucket in strata:
                if bucket:
                    queue.append(bucket.pop())
        queues[key] = queue
    keys = list(queues)
    keys = [keys[i] for i in rng.permutation(len(keys))]
    selected = []
    while len(selected) < config['n_series']:
        for key in keys:
            if queues[key]:
                selected.append(queues[key].pop(0))
            if len(selected) == config['n_series']:
                break
    return profile.loc[selected].sort_values('id').reset_index(drop=True)


def build_long(sales, calendar, prices, selected):
    wide = sales.loc[sales.id.isin(selected.id)]
    long = wide.melt(id_vars=IDS, value_vars=day_columns(sales), var_name='d', value_name='units_sold')
    long = long.merge(calendar, on='d', how='left', validate='many_to_one')
    if long.date.isna().any():
        raise ValueError('Sales days missing from calendar')
    long = long.merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left', validate='many_to_one')
    long['price_missing'] = long.sell_price.isna()
    long = long.rename(columns={'id': 'series_id'}).sort_values(['series_id', 'date']).reset_index(drop=True)
    if long.duplicated(['series_id', 'date']).any() or long.units_sold.isna().any() or long.units_sold.lt(0).any():
        raise ValueError('Invalid pilot sales keys/targets')
    if len(long) != len(selected) * len(day_columns(sales)):
        raise ValueError('Unexpected pilot row count')
    return long


def audit_data(calendar, sales, prices):
    days = day_columns(sales)
    assert not calendar.d.duplicated().any()
    assert not calendar.date.duplicated().any()
    assert calendar.date.sort_values().diff().dropna().eq(pd.Timedelta(days=1)).all()
    assert not sales.id.duplicated().any()
    assert not sales.duplicated(['item_id', 'store_id']).any()
    assert not prices.duplicated(['store_id', 'item_id', 'wm_yr_wk']).any()
    assert not sales[days].isna().any().any()
    assert sales[days].ge(0).all().all()
    assert prices.sell_price.gt(0).all()
    assert set(days).issubset(set(calendar.d))
    # Weekly join avoids expanding all 30,490 series to daily rows for price coverage.
    weeks = calendar.loc[calendar.d.isin(days), ['wm_yr_wk']].drop_duplicates()
    keys = sales[['item_id', 'store_id']].merge(weeks, how='cross')
    coverage = keys.merge(prices, how='left', on=['item_id', 'store_id', 'wm_yr_wk'], validate='one_to_one')
    summary = {
        'shapes': {name: list(df.shape) for name, df in [('calendar', calendar), ('sales', sales), ('prices', prices)]},
        'sales_days': len(days),
        'sales_start': str(calendar.set_index('d').loc[days[0], 'date'].date()),
        'sales_end': str(calendar.set_index('d').loc[days[-1], 'date'].date()),
        'calendar_end': str(calendar.date.max().date()),
        'items': int(sales.item_id.nunique()), 'stores': int(sales.store_id.nunique()),
        'states': sorted(sales.state_id.unique().tolist()), 'categories': sorted(sales.cat_id.unique().tolist()),
        'missing_values': {'calendar': calendar.isna().sum().to_dict(),
                           'sales_total': int(sales.isna().sum().sum()), 'prices': prices.isna().sum().to_dict()},
        'sales_zero_fraction': float(sales[days].eq(0).to_numpy().mean()),
        'weekly_price_coverage_fraction': float(coverage.sell_price.notna().mean()),
        'weekly_price_coverage_denominator': len(coverage),
        'snap_days': {c: int(calendar[c].sum()) for c in ['snap_CA', 'snap_TX', 'snap_WI']},
        'checks': 'Unique keys, continuous calendar, complete nonnegative sales, positive observed prices passed'
    }
    return summary


def plot_examples(long, selected, out):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    colors = ['#167d9a', '#c66b26', '#7955a2']
    for ax, segment, color in zip(axes, ['stable', 'volatile', 'intermittent'], colors):
        candidates = selected.loc[selected.demand_type.eq(segment), 'id']
        if candidates.empty:
            ax.set_title(f'No {segment} series selected'); continue
        sid = candidates.iloc[0]
        sample = long.loc[long.series_id.eq(sid)].head(730)
        ax.plot(sample.date, sample.units_sold, color=color, alpha=.35, linewidth=.7, label='Daily units')
        ax.plot(sample.date, sample.units_sold.rolling(28, min_periods=28).mean(), color=color, linewidth=1.7, label='28-day trailing mean')
        ax.set_title(f'{segment.title()} · {sid}', loc='left', fontsize=11)
        ax.set_ylabel('Units'); ax.grid(alpha=.15); ax.legend(loc='upper right', frameon=False)
    fig.suptitle('Pilot demand examples — historical sampling window only', fontsize=15)
    fig.savefig(out / 'representative_demand.png', dpi=160)
    plt.close(fig)
    table = pd.crosstab(selected.store_id, selected.demand_type).reindex(columns=['stable','volatile','intermittent'], fill_value=0)
    ax = table.plot.bar(stacked=True, color=colors, figsize=(10, 5), rot=0)
    ax.set(title='Pilot composition by store and demand type', xlabel='Store', ylabel='Selected series')
    ax.legend(title='Demand type', frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1))
    ax.figure.tight_layout(); ax.figure.savefig(out / 'pilot_composition.png', dpi=160); plt.close(ax.figure)


def run(root=ROOT):
    start = time.perf_counter()
    config = json.loads((root / 'configs/pilot.json').read_text())
    for folder in ['data/processed', 'results/metrics', 'results/figures']:
        (root / folder).mkdir(parents=True, exist_ok=True)
    calendar, sales, prices = load_data(root)
    summary = audit_data(calendar, sales, prices)
    profile = profile_series(sales, config)
    selected = select_pilot(profile, config)
    assert selected.id.tolist() == select_pilot(profile, config).id.tolist()
    long = build_long(sales, calendar, prices, selected)
    selected.to_csv(root / 'data/processed/pilot_series.csv', index=False)
    profile.to_csv(root / 'data/processed/series_profiles.csv', index=False)
    long.to_csv(root / 'data/processed/pilot_daily.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    selected.to_csv(root / 'results/metrics/pilot_manifest.csv', index=False)
    sales.groupby(['state_id', 'store_id', 'cat_id', 'dept_id']).size().rename('n_series').to_csv(root / 'results/metrics/hierarchy.csv')
    pd.concat([calendar.event_name_1, calendar.event_name_2]).dropna().value_counts().rename('calendar_occurrences').to_csv(root / 'results/metrics/events.csv')
    summary['pilot'] = {
        'series': len(selected), 'rows': len(long), 'profile_end': str(calendar.set_index('d').loc[f'd_{config["profile_days"]}', 'date'].date()),
        'excluded_inactive_series': int((~profile.eligible).sum()),
        'demand_types': selected.demand_type.value_counts().to_dict(),
        'volume_bands': selected.volume_band.value_counts().to_dict(),
        'stores': selected.store_id.value_counts().sort_index().to_dict(),
        'categories': selected.cat_id.value_counts().to_dict(),
        'missing_price_rows': int(long.price_missing.sum()),
        'positive_sales_missing_price_rows': int((long.price_missing & long.units_sold.gt(0)).sum())
    }
    (root / 'results/metrics/data_audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    plot_examples(long, selected, root / 'results/figures')
    hashes = {}
    for path in sorted((root / 'data/raw').glob('*.csv')):
        with path.open('rb') as f: hashes[path.name] = hashlib.file_digest(f, 'sha256').hexdigest()
    run_info = {'config': config, 'raw_sha256': hashes, 'python': sys.version, 'platform': platform.platform(),
                'versions': {'pandas': pd.__version__, 'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
                'elapsed_seconds': time.perf_counter() - start, 'models_run': False}
    (root / 'results/metrics/data_run.json').write_text(json.dumps(run_info, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    run(parser.parse_args().root)
