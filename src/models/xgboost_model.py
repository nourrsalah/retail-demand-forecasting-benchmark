"""Pooled sales-only XGBoost with history-safe features and recursive forecasts."""
import numpy as np
import xgboost as xgb


def feature_names(config):
    return ([f'lag_{lag}' for lag in config['lags']] +
            [f'rolling_{stat}_{window}' for window in config['rolling_windows']
             for stat in ['mean', 'std', 'zero_fraction']])


def lookback(config):
    values = config['lags'] + config['rolling_windows']
    if not values or any(not isinstance(v, int) or v < 1 for v in values):
        raise ValueError('Feature lookbacks must be positive integers')
    return max(values)


def validate_history(history, minimum):
    history = np.asarray(history, dtype=float)
    if history.ndim != 2 or not history.shape[0] or history.shape[1] < minimum:
        raise ValueError('Expected nonempty series-by-day history with sufficient observations')
    if not np.isfinite(history).all() or (history < 0).any():
        raise ValueError('History must contain finite nonnegative sales')
    return history


def features_from_history(history, config):
    """Each row ends strictly before the target being predicted."""
    history = validate_history(history, lookback(config))
    columns = [history[:, -lag] for lag in config['lags']]
    for window in config['rolling_windows']:
        recent = history[:, -window:]
        columns.extend([recent.mean(axis=1), recent.std(axis=1, ddof=0), (recent == 0).mean(axis=1)])
    return np.column_stack(columns).astype(np.float32)


def training_matrix(history, config):
    """Drop the first lookback targets; do not borrow observations before the allowed window."""
    width = lookback(config)
    history = validate_history(history, width + 1)
    blocks = np.lib.stride_tricks.sliding_window_view(history, width + 1, axis=1)
    past = blocks[..., :-1].reshape(-1, width)
    targets = blocks[..., -1].reshape(-1).astype(np.float32)
    return features_from_history(past, config), targets


def fit_model(features, targets, candidate, settings):
    params = {**settings['base_params'], 'max_depth': candidate['max_depth'],
              'seed': settings['seed'], 'nthread': settings['threads_per_fit']}
    train = xgb.DMatrix(features, label=targets, feature_names=feature_names(settings['features']),
                        nthread=settings['threads_per_fit'])
    # No eval_set, early stopping, or labels after the origin enter fitting.
    return xgb.train(params, train, num_boost_round=candidate['num_boost_round'])


def recursive_forecast(model, history, horizon, feature_config):
    """One rollout: later features use prior predictions, never realized future sales."""
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError('Horizon must be a positive integer')
    width = lookback(feature_config)
    rolling = validate_history(history, width)[:, -width:].copy()
    raw = np.empty((len(rolling), horizon), dtype=float)
    for step in range(horizon):
        features = features_from_history(rolling, feature_config)
        prediction = np.asarray(model.inplace_predict(features), dtype=float)
        if prediction.shape != (len(rolling),) or not np.isfinite(prediction).all():
            raise ValueError('Invalid pooled prediction shape or values')
        raw[:, step] = prediction
        rolling = np.column_stack([rolling[:, 1:], np.maximum(prediction, 0)])
    return raw
