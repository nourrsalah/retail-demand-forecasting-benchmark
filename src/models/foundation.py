"""Frozen CPU adapters; only historical target arrays cross this interface."""
import numpy as np


def validate_history(history, history_days=365):
    x = np.asarray(history, dtype=np.float32)
    if x.ndim != 2 or history_days not in [90,180,365] or x.shape[1] != history_days or not len(x):
        raise ValueError('Expected nonempty series × configured observed history')
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('History must contain finite nonnegative sales')
    return x


def validate_prediction(prediction, count, horizon):
    x = np.asarray(prediction, dtype=float)
    if x.shape != (count, horizon) or not np.isfinite(x).all():
        raise ValueError(f'Invalid forecast shape/values: {x.shape}')
    return x


class FoundationAdapter:
    def __init__(self, spec, checkpoint, settings):
        import torch
        self.spec, self.settings = spec, settings
        key = spec['id']
        if key == 'chronos2':
            from chronos import Chronos2Pipeline
            self.model = Chronos2Pipeline.from_pretrained(
                str(checkpoint), device_map='cpu', torch_dtype=torch.float32)
        elif key == 'timesfm25':
            import timesfm
            self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                str(checkpoint), torch_compile=False)
            self.model.compile(timesfm.ForecastConfig(
                max_context=((settings.get('history_days',365)+31)//32)*32, max_horizon=spec['max_horizon'],
                per_core_batch_size=settings['batch_size'], normalize_inputs=spec['normalize_inputs'],
                force_flip_invariance=spec['force_flip_invariance'], infer_is_positive=False,
                use_continuous_quantile_head=False))
        elif key == 'moirai11small':
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
            self.model = MoiraiForecast(module=MoiraiModule.from_pretrained(str(checkpoint)),
                prediction_length=28, context_length=settings.get('history_days',365), patch_size=spec['patch_size'],
                num_samples=spec['num_samples'], target_dim=1,
                feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
            self.model.eval()
            self.predictor = self.model.create_predictor(batch_size=settings['batch_size'], device='cpu')
        else:
            raise ValueError(f'Unknown frozen model: {key}')

    def predict(self, history, origin, horizon=28):
        import torch
        x = validate_history(history, self.settings.get('history_days',365))
        if horizon != 28:
            raise ValueError('Generate one 28-day forecast; score its prefixes outside the adapter')
        torch.manual_seed(self.settings['seed'])
        np.random.seed(self.settings['seed'])
        with torch.inference_mode():
            if self.spec['id'] == 'chronos2':
                _, points = self.model.predict_quantiles(
                    [torch.from_numpy(row.copy()) for row in x], prediction_length=horizon,
                    quantile_levels=[0.5], batch_size=self.settings['batch_size'],
                    context_length=self.settings.get('history_days',365), cross_learning=False)
                raw = np.stack([p.detach().cpu().numpy().reshape(horizon) for p in points])
            elif self.spec['id'] == 'timesfm25':
                # TimesFM's first returned array is channel 5 (median), not channel 0.
                _, distribution = self.model.forecast(horizon=horizon, inputs=[row for row in x])
                raw = distribution[..., 0]  # documented native mean head
            else:
                import pandas as pd
                from gluonts.dataset.common import ListDataset
                start = pd.Period(pd.Timestamp(origin) - pd.Timedelta(days=self.settings.get('history_days',365)-1), freq='D')
                dataset = ListDataset([{'start':start, 'target':row} for row in x], freq='D')
                forecasts = list(self.predictor.predict(dataset))
                expected_start = pd.Period(origin, freq='D') + 1
                if any(f.start_date != expected_start or f.samples.shape != (self.spec['num_samples'], horizon) for f in forecasts):
                    raise ValueError('Moirai forecast dates or sample count do not match the protocol')
                raw = np.stack([f.mean for f in forecasts])
        return validate_prediction(raw, len(x), horizon)
