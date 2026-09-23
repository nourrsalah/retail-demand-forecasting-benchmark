"""Protocol v1.0 point metrics; undefined scores are NaN with reasons."""
import numpy as np


def rmsse_scale(train):
    train = np.asarray(train, dtype=float)
    positive = np.flatnonzero(train > 0)
    if not len(positive):
        return np.nan, 'no_positive_training_demand'
    active = train[positive[0]:]
    if len(active) < 2:
        return np.nan, 'fewer_than_two_active_training_observations'
    scale = float(np.mean(np.diff(active) ** 2))
    if scale == 0:
        return np.nan, 'zero_training_difference_scale'
    return scale, ''


def score(train, actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.ndim != 1 or not len(actual):
        raise ValueError('Targets and predictions must be nonempty aligned vectors')
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('Nonfinite targets or predictions')
    error = actual - predicted
    scale, reason = rmsse_scale(train)
    total = float(actual.sum())
    return {
        'MAE': (float(np.abs(error).mean()), ''),
        'WAPE': (float(np.abs(error).sum() / total) if total else np.nan,
                 '' if total else 'zero_actual_demand'),
        'RMSSE': (float(np.sqrt(np.mean(error ** 2) / scale)) if np.isfinite(scale) else np.nan, reason)
    }


def pooled_wape(absolute_error_sum, actual_sum):
    """Pool numerator/denominator; never average series percentages."""
    denominator = float(np.sum(actual_sum))
    return float(np.sum(absolute_error_sum) / denominator) if denominator else np.nan


def equal_horizon_score(scores, horizons=(7, 14, 28)):
    """Undefined horizons invalidate the objective, without renormalization."""
    if set(scores) != set(horizons):
        raise ValueError('All configured horizons must be present exactly once')
    values = np.array([scores[h] for h in horizons], dtype=float)
    return float(values.mean()) if np.isfinite(values).all() else np.nan
