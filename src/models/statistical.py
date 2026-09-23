"""Small, explicit forecasting adapters. Inputs are training targets only."""
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel


class NonConvergenceError(RuntimeError):
    pass


def fit_model(y, specification, maxiter=300):
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Training targets must be finite, nonnegative, and one-dimensional')
    kind = specification['model']
    if kind == 'last_value':
        return ('last_value', y[-1:])
    if kind == 'weekly_seasonal_naive':
        if len(y) < 7:
            raise ValueError('Weekly baseline requires at least seven observations')
        return ('weekly_seasonal_naive', y[-7:].copy())
    if kind == 'ARIMA':
        fit = ARIMA(y, order=tuple(specification['order']),
                    seasonal_order=tuple(specification['seasonal_order']),
                    trend=specification['trend'], enforce_stationarity=True,
                    enforce_invertibility=True).fit(method='statespace',
                                                    method_kwargs={'maxiter': maxiter}, cov_type='none')
    elif kind == 'ETS':
        fit = ETSModel(y, error=specification['error'], trend=specification['trend'],
                       seasonal=specification['seasonal'], damped_trend=specification['damped_trend'],
                       seasonal_periods=specification.get('seasonal_periods'),
                       initialization_method='estimated').fit(maxiter=maxiter, disp=False)
    else:
        raise ValueError(f'Unknown model: {kind}')
    if not fit.mle_retvals.get('converged', False):
        raise NonConvergenceError('Optimizer did not converge within the prespecified budget')
    return fit


def predict(fitted, horizon):
    if isinstance(fitted, tuple):
        raw = np.resize(fitted[1], horizon)
    else:
        raw = np.asarray(fitted.forecast(horizon), dtype=float)
    if raw.shape != (horizon,) or not np.isfinite(raw).all():
        raise ValueError('Invalid forecast shape or nonfinite predictions')
    return raw
