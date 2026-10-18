"""Known-calendar features and origin-known price; no realized future prices."""
import numpy as np
import pandas as pd
import xgboost as xgb
from src.models.xgboost_model import training_matrix, features_from_history, lookback, feature_names

EXTRA_NAMES=['weekday_sin','weekday_cos','year_sin','year_cos','scheduled_event','scheduled_snap','previous_price','previous_price_missing']


def business_features(history, future, state):
    """History ends at origin; only known calendar columns are read from future."""
    def calendar(frame):
        dates=pd.DatetimeIndex(frame.date)
        return np.column_stack([np.sin(2*np.pi*dates.dayofweek/7),np.cos(2*np.pi*dates.dayofweek/7),
            np.sin(2*np.pi*(dates.dayofyear-1)/365.25),np.cos(2*np.pi*(dates.dayofyear-1)/365.25),
            (frame.event_name_1.notna()|frame.event_name_2.notna()).astype(float),frame[f'snap_{state}'].to_numpy()])
    price=pd.to_numeric(history.sell_price,errors='coerce').ffill()
    lag=price.shift(1)
    historical=np.column_stack([calendar(history),lag.fillna(0),lag.isna().astype(float)])
    last=price.iloc[-1]
    future_extra=np.column_stack([calendar(future),np.full(len(future),0 if pd.isna(last) else last),np.full(len(future),float(pd.isna(last)))])
    return historical.astype(np.float32),future_extra.astype(np.float32)


def fit_business(history, extras, spec, settings):
    features,targets=training_matrix(history,settings['features']);width=lookback(settings['features'])
    features=np.column_stack([features,extras[:,width:,:].reshape(-1,extras.shape[-1])])
    params={**settings['base_params'],'max_depth':spec['max_depth'],'seed':settings['seed'],'nthread':settings['threads_per_fit']}
    data=xgb.DMatrix(features,label=targets,feature_names=feature_names(settings['features'])+EXTRA_NAMES,nthread=settings['threads_per_fit'])
    return xgb.train(params,data,num_boost_round=spec['num_boost_round'])


def predict_business(model,history,future_extras,settings):
    width=lookback(settings['features']);rolling=history[:,-width:].copy();raw=np.empty((len(history),len(future_extras[0])))
    for lead in range(raw.shape[1]):
        x=np.column_stack([features_from_history(rolling,settings['features']),future_extras[:,lead,:]])
        y=np.asarray(model.inplace_predict(x),dtype=float)
        if y.shape!=(len(history),) or not np.isfinite(y).all():raise ValueError('Invalid business-feature forecast')
        raw[:,lead]=y;rolling=np.column_stack([rolling[:,1:],np.maximum(y,0)])
    return raw
