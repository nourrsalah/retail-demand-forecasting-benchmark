"""Matched validation comparison with explicit native point estimators."""
import argparse
import json
from pathlib import Path
import pandas as pd
from src.evaluation.stage4 import ROOT, consistency, sha
from src.evaluation.report_stage5 import verify_comparable, matched_cases, markdown
from src.evaluation.metrics import pooled_wape


def write_report(run_dir):
    run_dir = Path(run_dir).resolve(); info = json.loads((run_dir/'run.json').read_text())
    if info['state'] != 'complete' or info['smoke'] or set(info['models_run']) != {s['id'] for s in info['settings']['models']}:
        raise ValueError('A completed full three-model validation run is required')
    frames = []; choices = {}; sources = {}
    for stage in [4,5]:
        folder = ROOT/'results/metrics'/info['settings'][f'stage{stage}_reference_run']
        previous = json.loads((folder/'run.json').read_text()); verify_comparable(previous,info)
        selected = json.loads((folder/'selected_models.json').read_text())['selected_candidates']
        frame = pd.read_csv(folder/'case_scores.csv'); frame = frame.loc[frame.candidate.isin(selected.values())].copy()
        frame['point_estimator'] = 'local_point_forecast'; frame['mean_comparison_eligible'] = True
        frames.append(frame); choices.update(selected); sources[str(folder.relative_to(ROOT))] = sha(folder/'case_scores.csv')
    frames.append(pd.read_csv(run_dir/'case_scores.csv'))
    choices.update({s['model']:s['id'] for s in info['settings']['models']})
    combined = pd.concat(frames,ignore_index=True)
    if combined.groupby(['series_id','forecast_origin','horizon']).actual_sum.nunique().gt(1).any():
        raise ValueError('Forecast targets disagree')
    tables = {}
    for label,frame in [('native_estimators',combined),('mean_only',combined.loc[combined.mean_comparison_eligible])]:
        candidates = sorted(frame.candidate.unique()); matched = matched_cases(frame,candidates)
        rows = []
        for (model,candidate,estimator,h),g in matched.groupby(['model','candidate','point_estimator','horizon']):
            rows.append({'model':model,'candidate':candidate,'point_estimator':estimator,'horizon':h,
                         'common_cases':len(g),'MAE':g.MAE.mean(),'RMSSE':g.RMSSE.mean(),
                         'WAPE':pooled_wape(g.absolute_error_sum,g.actual_sum)})
        table = pd.DataFrame(rows,columns=['model','candidate','point_estimator','horizon','common_cases','MAE','RMSSE','WAPE'])
        table.to_csv(run_dir/f'comparison_{label}.csv',index=False); tables[label] = table
        eligible_choices = {m:c for m,c in choices.items() if c in candidates}
        reliability,wins = consistency(frame,eligible_choices,len(info['actual_origins']))
        reliability.to_csv(run_dir/f'reliability_{label}.csv',index=False)
        wins.to_csv(run_dir/f'win_rates_{label}.csv',index=False)
    combined.groupby(['model','candidate','horizon']).agg(scheduled_cases=('status','size'),
        successful_cases=('status',lambda x:int(x.eq('ok').sum())),rmsse_eligible=('RMSSE','count')).to_csv(run_dir/'combined_coverage.csv')
    native = tables['native_estimators']
    aggregate = native.groupby(['model','point_estimator'])[['MAE','RMSSE','WAPE']].mean().reset_index()
    for col in ['MAE','RMSSE','WAPE']: aggregate[col] = aggregate[col].map(lambda x:f'{x:.4f}')
    logs = pd.read_csv(run_dir/'resources.csv')
    for column in ['inference_seconds','sampled_peak_rss_bytes']:
        if column not in logs: logs[column] = float('nan')
    resources = logs.groupby('candidate').agg(failed_origins=('status',lambda s:int(s.ne('ok').sum())),
        first_inference_seconds=('inference_seconds','sum'),peak_worker_RSS_bytes=('sampled_peak_rss_bytes','max')).reset_index()
    resources['peak_worker_RSS_MiB'] = (resources.pop('peak_worker_RSS_bytes')/1024**2).round(1)
    resources['first_inference_seconds'] = resources.first_inference_seconds.round(2)
    text = f'''# Stage 6 validation results — pretrained models

Run: `{info['run_id']}`. **Validation only; final test periods remain unevaluated.**

Three prespecified frozen checkpoints, 200 series, three origins, 365 observed daily sales values per forecast. No fitting, future labels or business covariates. {info['failed_series_cases']} failures out of {info['scheduled_series_cases']} scheduled model/series/origin cases. Full wall time including three warm repeats: {info['elapsed_wall_seconds']:.1f} seconds.

## Native-estimator comparison

The following table explicitly mixes native point estimators. Chronos-2 supplies a **median**, while TimesFM supplies its mean head and Moirai supplies a 200-sample mean. It is an operational comparison, not a controlled all-means ranking. Each number gives equal 1/3 weight to the 7/14/28-day aggregates. WAPE is pooled within each horizon.

{markdown(aggregate) if len(aggregate) else 'No common valid support across all models; see coverage and per-model scores.'}

Common RMSSE-eligible support per horizon: {native.common_cases.min() if len(native) else 0}–{native.common_cases.max() if len(native) else 0} out of 600. Per-model coverage and failures are retained in `combined_coverage.csv`. A separate `comparison_mean_only.csv` excludes Chronos-2; its win rates are in `win_rates_mean_only.csv`. The naive forecasts remain reference rules. Existing ARIMA/ETS/XGBoost choices were reused, not retuned.

## Runtime and memory

{markdown(resources)}

All models run sequentially on the recorded CPU host. First inference is summed across origins; loading and three warmed repeats are separate in `resources.csv`. RSS includes the full worker runtime, sampled every 10 ms; brief peaks can be missed. Checkpoint downloads and model initialization are not training time. Moirai's selected checkpoint has a noncommercial license.

## Limits and saved evidence

- `accuracy_by_horizon.csv`, `case_scores.csv`, `metrics.csv`: per-model accuracy, eligibility and failures.
- `reliability_mean_only.csv` / `reliability_native_estimators.csv`: across-series median, p90, p95, spread and maximum; complete origins required.
- `origin_distributions.csv`: origin-level distributions.
- `candidate_selection.csv`: equal-horizon validation objective, one prespecified configuration per model.
- `run.json`, checkpoint lock and environment files: data/source hashes, pinned revisions, settings and hardware.
- `results/forecasts/{info['run_id']}/forecasts.csv.gz`: raw and clipped forecasts.

Moirai’s published family pretraining corpus includes M5. No M5-disjoint pretraining guarantee has been established for these checkpoints; see `docs/pretraining_audit.md`. These scores cannot demonstrate clean unseen-data generalization or a universal model-family winner. Pilot selection excludes cold starts. History sensitivity, business features, uncertainty intervals and the frozen final evaluation remain later work.
'''
    (ROOT/'docs/stage6_results.md').write_text(text); (run_dir/'report.md').write_text(text)
    (run_dir/'comparison_provenance.json').write_text(json.dumps({'reference_case_scores_sha256':sources,
        'stage6_case_scores_sha256':sha(run_dir/'case_scores.csv'),'report_source_sha256':sha(Path(__file__))},indent=2)+'\n')
    print(aggregate.to_string(index=False));print('Report:',ROOT/'docs/stage6_results.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run_dir',type=Path)
    write_report(parser.parse_args().run_dir)
