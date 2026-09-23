"""Run Stage 4 on VALIDATION ONLY. No final-test execution path is exposed."""
import os
for _key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[_key] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import signal
import subprocess
import threading
import time
import warnings

import numpy as np
import pandas as pd
import psutil
import scipy
import statsmodels
from src.models.statistical import fit_model, predict
from src.evaluation.metrics import score, rmsse_scale, pooled_wape, equal_horizon_score
from src.evaluation.windows import validation_data, make_window

ROOT = Path(__file__).resolve().parents[2]
CASE = ['series_id', 'forecast_origin']


def _timeout(signum, frame):
    raise TimeoutError('Fit/forecast exceeded configured wall-time limit')


def run_task(task):
    """Worker receives history only, never validation or test actuals."""
    spec, y, settings = task['spec'], task['train'], task['settings']
    process = psutil.Process()
    rss = [process.memory_info().rss]
    stop = threading.Event()
    def sample():
        while not stop.wait(settings['rss_sample_interval_seconds']):
            rss.append(process.memory_info().rss)
    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    result = {k: task[k] for k in ['candidate', 'series_id', 'forecast_origin']}
    result.update(status='ok', error='', fit_seconds=np.nan, inference_seconds=np.nan,
                  warm_inference_median_seconds=np.nan, warm_inference_range_seconds=np.nan,
                  raw=None, negative_raw_count=0)
    captured = []
    start = time.perf_counter()
    signal.signal(signal.SIGALRM, _timeout)
    signal.setitimer(signal.ITIMER_REAL, settings['fit_forecast_timeout_seconds'])
    try:
        with warnings.catch_warnings(record=True) as caught:
            captured = caught
            warnings.simplefilter('always')
            fitted = fit_model(y, spec, settings['maxiter'])
            result['fit_seconds'] = time.perf_counter() - start
            tic = time.perf_counter()
            raw = predict(fitted, task['horizon'])
            result['inference_seconds'] = time.perf_counter() - tic
            timings = []
            for _ in range(settings['warm_inference_repeats']):
                tic = time.perf_counter()
                repeated = predict(fitted, task['horizon'])
                timings.append(time.perf_counter() - tic)
                if not np.allclose(raw, repeated, rtol=1e-12, atol=1e-12):
                    raise ValueError('Repeated point forecasts changed unexpectedly')
            result.update(raw=raw.tolist(), negative_raw_count=int((raw < 0).sum()),
                          warm_inference_median_seconds=float(np.median(timings)),
                          warm_inference_range_seconds=float(np.ptp(timings)))
    except Exception as exc:
        result.update(status='timeout' if isinstance(exc, TimeoutError) else 'failed',
                      error=f'{type(exc).__name__}: {exc}', raw=None)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        stop.set(); monitor.join()
    result.update(elapsed_seconds=time.perf_counter()-start,
                  sampled_peak_rss_bytes=max(rss + [process.memory_info().rss]),
                  warnings=' | '.join(sorted(set(str(w.message) for w in captured))))
    return result


def select_candidates(cases, candidates, horizons):
    """Same successful cases per family; never reward dropping difficult series."""
    rows, selections = [], {}
    for model in sorted({c['model'] for c in candidates}):
        specs = [c for c in candidates if c['model'] == model]
        ids = [c['id'] for c in specs]
        group = cases.loc[cases.candidate.isin(ids)]
        good = group.loc[group.status.eq('ok')].groupby(CASE).size()
        common_keys = good[good.eq(len(ids) * len(horizons))].index
        common = group.set_index(CASE).loc[common_keys].reset_index() if len(common_keys) else group.iloc[:0]
        all_scales_undefined = not group.scale_valid.any()
        objective_metric = 'MAE' if all_scales_undefined else 'RMSSE'
        objective_rows = common if all_scales_undefined else common.loc[common.scale_valid]
        for spec in specs:
            candidate = spec['id']
            frame = objective_rows.loc[objective_rows.candidate.eq(candidate)]
            mae_frame = common.loc[common.candidate.eq(candidate)]
            values = {h: frame.loc[frame.horizon.eq(h), objective_metric].mean() for h in horizons}
            maes = {h: mae_frame.loc[mae_frame.horizon.eq(h), 'MAE'].mean() for h in horizons}
            own = group.loc[group.candidate.eq(candidate)]
            n_runs = len(own[CASE].drop_duplicates())
            failed = len(own.loc[own.status.ne('ok'), CASE].drop_duplicates())
            rows.append({'model': model, 'candidate': candidate, 'selection_metric': objective_metric,
                         'selection_score': equal_horizon_score(values, horizons),
                         'mae_tie_break': equal_horizon_score(maes, horizons),
                         'common_successful_cases': len(common_keys),
                         'common_scored_cases': len(frame[CASE].drop_duplicates()),
                         'scheduled_cases': n_runs, 'failed_cases': failed,
                         'failure_rate': failed/n_runs, **{f'score_h{h}': values[h] for h in horizons}})
        ranking = [r for r in rows if r['model'] == model and np.isfinite(r['selection_score'])]
        ranking.sort(key=lambda r: (r['selection_score'], r['mae_tie_break'], r['candidate']))
        selections[model] = ranking[0]['candidate'] if ranking else None
    return pd.DataFrame(rows), selections


def summarize(cases, logs):
    rows = []
    for (candidate, horizon), group in cases.groupby(['candidate', 'horizon']):
        valid = group.loc[group.status.eq('ok')]
        rows.append({'candidate': candidate, 'horizon': horizon,
                     'scheduled_cases': len(group), 'valid_cases': len(valid),
                     'valid_rmsse_cases': int(valid.RMSSE.notna().sum()),
                     'MAE': valid.MAE.mean(), 'RMSSE': valid.RMSSE.mean(),
                     'WAPE': pooled_wape(valid.absolute_error_sum, valid.actual_sum)})
    resource = logs.groupby('candidate').agg(
        scheduled_cases=('status','size'), failed_cases=('status',lambda s:int(s.ne('ok').sum())),
        fit_seconds_total=('fit_seconds','sum'), inference_seconds_total=('inference_seconds','sum'),
        elapsed_worker_seconds=('elapsed_seconds','sum'),
        peak_sampled_worker_rss_bytes=('sampled_peak_rss_bytes','max'),
        negative_raw_values=('negative_raw_count','sum')).reset_index()
    return pd.DataFrame(rows), resource


def consistency(cases, selected, n_origins):
    chosen = cases.loc[cases.candidate.isin([v for v in selected.values() if v])]
    rows = []
    for (candidate, horizon), g in chosen.groupby(['candidate','horizon']):
        for metric in ['MAE','RMSSE']:
            valid = g.loc[g.status.eq('ok') & g[metric].notna()]
            agg = valid.groupby('series_id')[metric].agg(['mean','count'])
            values = agg.loc[agg['count'].eq(n_origins), 'mean']
            rows.append({'candidate':candidate,'horizon':horizon,'metric':metric,'n_series':len(values),
                         'excluded_series':g.series_id.nunique()-len(values),
                         'median':values.median(),'p90':values.quantile(.90),'p95':values.quantile(.95),
                         'std':values.std(ddof=0),'IQR':values.quantile(.75)-values.quantile(.25),
                         'maximum':values.max(),'worst_series':values.idxmax() if len(values) else ''})
    wins=[]
    for horizon in sorted(chosen.horizon.unique()):
        for metric in ['RMSSE','MAE']:
            g=chosen.loc[chosen.horizon.eq(horizon)]
            agg=g.groupby(['series_id','candidate'])[metric].agg(['mean','count'])
            complete=agg.loc[agg['count'].eq(n_origins),'mean'].unstack('candidate')
            complete=complete.reindex(columns=sorted(chosen.candidate.unique())).dropna()
            if len(complete):
                ties=np.isclose(complete.to_numpy(),complete.min(axis=1).to_numpy()[:,None],atol=1e-8,rtol=0)
                rates=(ties/ties.sum(axis=1)[:,None]).mean(axis=0)
            else:
                rates=np.full(len(complete.columns), np.nan)
            for candidate, rate in zip(complete.columns,rates):
                wins.append({'candidate':candidate,'horizon':horizon,'metric':metric,
                             'win_rate':rate,'n_common_series':len(complete),
                             'excluded_series':g.series_id.nunique()-len(complete)})
    return pd.DataFrame(rows),pd.DataFrame(wins)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def run(root=ROOT, smoke=False, workers=None):
    started = time.perf_counter()
    protocol_path=root/'configs/evaluation.json'
    stage_path=root/'configs/stage4.json'
    protocol=json.loads(protocol_path.read_text()); settings=json.loads(stage_path.read_text())
    if settings['split']!='validation' or settings['scenario']!='univariate':
        raise ValueError('Stage 4 runner permits univariate validation only')
    if settings['history_days']!=protocol['primary_history_days']:
        raise ValueError('This run must use locked primary history')
    horizons=protocol['horizons_days']
    weights=protocol['validation_model_selection']['horizon_weights']
    if any(not np.isclose(weights[str(h)],1/len(horizons)) for h in horizons):
        raise ValueError('Runner requires the locked equal horizon weights')
    manifest=pd.read_csv(root/protocol['pilot_manifest'])
    n_workers=workers or settings['workers']
    if smoke:
        # Predetermined smoke subset: first ID in each frozen demand segment.
        manifest=manifest.sort_values('id').groupby('demand_type',sort=True).head(1)
    pilot=pd.read_csv(root/'data/processed/pilot_daily.csv.gz',usecols=['series_id','date','units_sold'],parse_dates=['date'])
    pilot=validation_data(pilot,protocol)
    pilot=pilot.loc[pilot.series_id.isin(manifest.id)]
    if set(pilot.series_id)!=set(manifest.id):
        raise ValueError('Pilot and frozen manifest disagree')
    origins=protocol['validation_origins'][:1] if smoke else protocol['validation_origins']
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+('-stage4-smoke' if smoke else '-stage4-validation')
    out=root/'results/metrics'/run_id; out.mkdir(parents=True,exist_ok=False)
    forecast_dir=root/'results/forecasts'/run_id; forecast_dir.mkdir(parents=True,exist_ok=False)
    tasks=[]; targets={}; trains={}
    for sid,g in pilot.groupby('series_id',sort=True):
        series=g.set_index('date').units_sold.sort_index()
        for origin in origins:
            train,actual,dates=make_window(series,origin,settings['history_days'],max(horizons))
            targets[(sid,origin)]=(actual,dates); trains[(sid,origin)]=train
            for spec in settings['candidates']:
                tasks.append({'spec':spec,'candidate':spec['id'],'series_id':sid,'forecast_origin':origin,
                              'train':train,'horizon':max(horizons),'settings':settings})
    source_paths=[protocol_path,stage_path,root/'docs/evaluation_protocol.md',root/'requirements.txt',
                  root/protocol['pilot_manifest'],root/'data/processed/pilot_daily.csv.gz']
    source_paths += sorted((root/'src/models').glob('*.py'))+sorted((root/'src/evaluation').glob('*.py'))
    tracked_hashes={str(p.relative_to(root)):sha(p) for p in source_paths}
    revision=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True).stdout.strip()
    info={'run_id':run_id,'state':'running','split':'validation','smoke':smoke,
          'started_utc':datetime.now(timezone.utc).isoformat(),'settings':settings,'protocol':protocol,
          'actual_origins':origins,'series_count':len(manifest),'workers':n_workers,'scheduled_tasks':len(tasks),
          'files_sha256':tracked_hashes,'git_revision':revision,'source_hashes_capture_uncommitted_changes':True,
          'python':platform.python_version(),'platform':platform.platform(),'machine':platform.machine(),
          'logical_cpus':os.cpu_count(),'host_ram_bytes':psutil.virtual_memory().total,
          'hardware':'CPU only','blas_threads_per_worker':1,
          'packages':{'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,
                      'statsmodels':statsmodels.__version__,'psutil':psutil.__version__},
          'memory_measurement':'10ms sampled RSS per worker job, includes worker runtime; can miss brief peaks; not aggregate host peak',
          'timeout_measurement':'POSIX interval timer; delivery may wait for native code to return',
          'setup_download_seconds':None,'gpu_memory_bytes':None}
    (out/'run.json').write_text(json.dumps(info,indent=2)+'\n')
    results=[]
    print(f'{run_id}: {len(tasks)} validation tasks, {len(manifest)} series, {n_workers} workers',flush=True)
    with ProcessPoolExecutor(max_workers=n_workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(run_task,t) for t in tasks]
        with (out/'task_log.jsonl').open('w') as logfile:
            for future in as_completed(futures):
                result=future.result(); results.append(result)
                logfile.write(json.dumps({k:v for k,v in result.items() if k!='raw'})+'\n'); logfile.flush()
                if len(results)%100==0 or len(results)==len(tasks):
                    print(f'Completed {len(results)}/{len(tasks)}; failures {sum(r["status"]!="ok" for r in results)}',flush=True)
    forecasts=[]; cases=[]; metrics=[]
    specifications={s['id']:s for s in settings['candidates']}
    for result in sorted(results,key=lambda r:(r['candidate'],r['series_id'],r['forecast_origin'])):
        sid,origin=result['series_id'],result['forecast_origin']
        actual,dates=targets[(sid,origin)]; train=trains[(sid,origin)]
        spec=specifications[result['candidate']]
        raw=np.asarray(result['raw']) if result['raw'] is not None else np.full(max(horizons),np.nan)
        predicted=np.maximum(raw,0)
        common={'run_id':run_id,'model':spec['model'],'model_version':statsmodels.__version__ if spec['model'] in ['ARIMA','ETS'] else '1',
                'candidate':spec['id'],'scenario':'univariate','history_window':settings['history_days'],
                'series_id':sid,'forecast_origin':origin,'status':result['status']}
        scale,scale_reason=rmsse_scale(train)
        for horizon in horizons:
            fields={**common,'horizon':horizon}
            values=score(train,actual[:horizon],predicted[:horizon]) if result['status']=='ok' else {m:(np.nan,result['error']) for m in ['MAE','WAPE','RMSSE']}
            for metric,(value,reason) in values.items():
                metrics.append({**fields,'metric':metric,'value':value,'valid':bool(np.isfinite(value)),
                                'reason':reason,'n_observations':horizon})
            cases.append({**fields,**{m:v[0] for m,v in values.items()},'scale_valid':bool(np.isfinite(scale)),
                          'scale_reason':scale_reason,'actual_sum':float(actual[:horizon].sum()),
                          'absolute_error_sum':float(np.abs(actual[:horizon]-predicted[:horizon]).sum())})
            for i in range(horizon):
                forecasts.append({**fields,'lead':i+1,'date':str(dates[i].date()),'y_true':actual[i],
                                  'y_pred_raw':raw[i],'y_pred':predicted[i]})
    case_df=pd.DataFrame(cases); log_df=pd.DataFrame([{k:v for k,v in r.items() if k!='raw'} for r in results])
    forecast_df=pd.DataFrame(forecasts)
    key=['run_id','candidate','scenario','history_window','series_id','forecast_origin','horizon','lead']
    assert not forecast_df.duplicated(key).any()
    assert set(forecast_df.forecast_origin)==set(origins)
    assert forecast_df.date.max()<=str((pd.Timestamp(protocol['validation_origins'][-1])+pd.Timedelta(days=max(horizons))).date())
    forecast_df.to_csv(forecast_dir/'candidate_forecasts.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    case_df.to_csv(out/'case_scores.csv',index=False); log_df.to_csv(out/'resources.csv',index=False)
    summary,resource_summary=summarize(case_df,log_df)
    summary.to_csv(out/'accuracy_by_horizon.csv',index=False); resource_summary.to_csv(out/'resource_summary.csv',index=False)
    ranking,selections=select_candidates(case_df,settings['candidates'],horizons)
    ranking.to_csv(out/'candidate_selection.csv',index=False)
    selected_forecasts=forecast_df.loc[forecast_df.candidate.isin([c for c in selections.values() if c])]
    standard_key=['run_id','model','scenario','history_window','series_id','forecast_origin','horizon','lead']
    assert not selected_forecasts.duplicated(standard_key).any()
    selected_forecasts.to_csv(forecast_dir/'forecasts.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    reliability,wins=consistency(case_df,selections,len(origins))
    reliability.to_csv(out/'reliability.csv',index=False); wins.to_csv(out/'win_rates.csv',index=False)
    case_df.groupby(['candidate','forecast_origin','horizon'])[['MAE','RMSSE']].agg(['median',lambda x:x.quantile(.9),lambda x:x.quantile(.95)]).to_csv(out/'origin_distributions.csv')
    frozen={'run_id':run_id,'status':'smoke_only_not_selected' if smoke else 'validation_selected',
            'scenario':'univariate','history_days':settings['history_days'],'selected_candidates':selections,
            'candidate_specs':{m:specifications[c] if c else None for m,c in selections.items()},
            'selection_config_sha256':sha(stage_path),'protocol_sha256':sha(protocol_path),
            'test_evaluated':False}
    (out/'selected_models.json').write_text(json.dumps(frozen,indent=2)+'\n')
    info.update(state='complete',elapsed_wall_seconds=time.perf_counter()-started,forecast_rows=len(forecast_df),
                failed_tasks=int(log_df.status.ne('ok').sum()),final_test_evaluated=False)
    (out/'run.json').write_text(json.dumps(info,indent=2)+'\n')
    print(ranking.to_string(index=False),flush=True)
    print(f'Results: {out}',flush=True)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true',help='One frozen example per demand segment, first validation origin')
    parser.add_argument('--workers',type=int,default=None)
    args=parser.parse_args()
    run(smoke=args.smoke,workers=args.workers)
