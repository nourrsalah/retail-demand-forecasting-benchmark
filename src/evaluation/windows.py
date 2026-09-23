"""Date-exact splits: no future actuals are supplied to model adapters."""
import numpy as np
import pandas as pd


def validation_data(pilot, protocol):
    cutoff = pd.Timestamp(protocol['validation_origins'][-1]) + pd.Timedelta(days=max(protocol['horizons_days']))
    return pilot.loc[pilot.date.le(cutoff), ['series_id', 'date', 'units_sold']].copy()


def make_window(series, origin, history, horizon):
    origin = pd.Timestamp(origin)
    training_dates = pd.date_range(end=origin, periods=history, freq='D')
    target_dates = pd.date_range(start=origin + pd.Timedelta(days=1), periods=horizon, freq='D')
    if not series.index.is_unique:
        raise ValueError('Duplicate dates')
    train, actual = series.reindex(training_dates), series.reindex(target_dates)
    if train.isna().any() or actual.isna().any():
        raise ValueError('Incomplete daily training or target window')
    if (train < 0).any() or (actual < 0).any():
        raise ValueError('Negative unit sales')
    return train.to_numpy(dtype=float), actual.to_numpy(dtype=float), target_dates
