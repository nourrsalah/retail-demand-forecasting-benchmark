"""Pooled XGBoost on locked validation origins; no test execution option."""
import os
for _name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[_name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
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
import xgboost as xgb
from src.models.xgboost_model import training_matrix, fit_model, recursive_forecast, lookback, feature_names
from src.evaluation.metrics import score, rmsse_scale
from src.evaluation.windows import make_window, validation_data
from src.evaluation.stage4 import ROOT, sha, select_candidates, summarize, consistency, _timeout


def run_pooled_task(task):
    settings=task['settings']; history=task['history']
    process=psutil.Process(); samples=[process.memory_info().rss]; stop=threading.Event()
    def sample():
        while not stop.wait(settings['rss_sample_interval_seconds']):
            samples.append(process.memory_info().rss)
    monitor=threading.Thread(target=sample,daemon=True); monitor.start()
    result={'candidate':task['spec']['id'],'forecast_origin':task['forecast_origin'],
            'status':'ok','error':'','feature_seconds':np.nan,'fit_seconds':np.nan,
            'inference_seconds':np.nan,'warm_inference_median_seconds':np.nan,
            'warm_inference_range_seconds':np.nan,'training_rows':0,'feature_count':0,
            'raw':None,'negative_raw_count':0,'feature_gain':{}}
    captured=[]; start=time.perf_counter()
    signal.signal(signal.SIGALRM,_timeout)
    signal.setitimer(signal.ITIMER_REAL,settings['fit_forecast_timeout_seconds'])
    try:
        with warnings.catch_warnings(record=True) as caught:
            captured=caught; warnings.simplefilter('always')
            tic=time.perf_counter()
            features,targets=training_matrix(history,settings['features'])
            result.update(feature_seconds=time.perf_counter()-tic,training_rows=len(targets),feature_count=features.shape[1])
            tic=time.perf_counter(); model=fit_model(features,targets,task['spec'],settings)
            result['fit_seconds']=time.perf_counter()-tic
            tic=time.perf_counter(); raw=recursive_forecast(model,history,task['horizon'],settings['features'])
            result['inference_seconds']=time.perf_counter()-tic
            timings=[]
            for _ in range(settings['warm_inference_repeats']):
                tic=time.perf_counter(); repeated=recursive_forecast(model,history,task['horizon'],settings['features'])
                timings.append(time.perf_counter()-tic)
                if not np.allclose(raw,repeated,rtol=1e-12,atol=1e-12):
                    raise ValueError('Repeated forecasts changed')
            result.update(raw=raw.tolist(),negative_raw_count=int((raw<0).sum()),
                          warm_inference_median_seconds=float(np.median(timings)),
                          warm_inference_range_seconds=float(np.ptp(timings)),
                          feature_gain=model.get_score(importance_type='gain'))
    except Exception as exc:
        result.update(status='timeout' if isinstance(exc,TimeoutError) else 'failed',
                      error=f'{type(exc).__name__}: {exc}',raw=None)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);stop.set();monitor.join()
    result.update(elapsed_seconds=time.perf_counter()-start,
                  sampled_peak_rss_bytes=max(samples+[process.memory_info().rss]),
                  warnings=' | '.join(sorted(set(str(w.message) for w in captured))))
    return result


def run(root=ROOT,smoke=False):
    start=time.perf_counter()
    protocol_path=root/'configs/evaluation.json'; config_path=root/'configs/stage5.json'
    protocol=json.loads(protocol_path.read_text()); settings=json.loads(config_path.read_text())
    if settings['split']!='validation' or settings['scenario']!='univariate':
        raise ValueError('Stage 5 permits univariate validation only')
    if settings['history_days']!=protocol['primary_history_days']:
        raise ValueError('Primary history mismatch')
    horizons=protocol['horizons_days']
    if any(not np.isclose(protocol['validation_model_selection']['horizon_weights'][str(h)],1/len(horizons)) for h in horizons):
        raise ValueError('Equal horizon weights required')
    manifest=pd.read_csv(root/protocol['pilot_manifest'])
    if smoke:manifest=manifest.sort_values('id').groupby('demand_type',sort=True).head(1)
    ids=sorted(manifest.id.tolist());origins=protocol['validation_origins'][:1] if smoke else protocol['validation_origins']
    data=pd.read_csv(root/'data/processed/pilot_daily.csv.gz',usecols=['series_id','date','units_sold'],parse_dates=['date'])
    data=validation_data(data,protocol)
    data=data.loc[data.series_id.isin(ids)]
    if set(data.series_id)!=set(ids):raise ValueError('Pilot membership mismatch')
    series={sid:g.set_index('date').units_sold.sort_index() for sid,g in data.groupby('series_id')}
    histories={}; actuals={}; dates={}
    for origin in origins:
        histories[origin]=[]
        for sid in ids:
            train,actual,target_dates=make_window(series[sid],origin,settings['history_days'],max(horizons))
            histories[origin].append(train);actuals[(sid,origin)]=actual;dates[origin]=target_dates
        histories[origin]=np.stack(histories[origin])
    tasks=[{'spec':spec,'history':histories[origin],'forecast_origin':origin,
            'horizon':max(horizons),'settings':settings} for origin in origins for spec in settings['candidates']]
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+('-stage5-smoke' if smoke else '-stage5-validation')
    out=root/'results/metrics'/run_id;out.mkdir(parents=True,exist_ok=False)
    forecasts_dir=root/'results/forecasts'/run_id;forecasts_dir.mkdir(parents=True,exist_ok=False)
    paths=[protocol_path,config_path,root/'docs/evaluation_protocol.md',root/'docs/stage5.md',root/'requirements.txt',
           root/protocol['pilot_manifest'],root/'data/processed/pilot_daily.csv.gz']
    paths+=sorted((root/'src/models').glob('*.py'))+sorted((root/'src/evaluation').glob('*.py'))
    info={'run_id':run_id,'state':'running','split':'validation','smoke':smoke,'settings':settings,'protocol':protocol,
          'actual_origins':origins,'series_ids':ids,'series_count':len(ids),'scheduled_pooled_jobs':len(tasks),
          'scheduled_series_cases':len(tasks)*len(ids),'forecast_strategy':settings['forecast_strategy'],
          'started_utc':datetime.now(timezone.utc).isoformat(),
          'files_sha256':{str(p.relative_to(root)):sha(p) for p in paths},
          'git_revision':subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True).stdout.strip(),
          'python':platform.python_version(),'platform':platform.platform(),'machine':platform.machine(),
          'logical_cpus':os.cpu_count(),'host_ram_bytes':psutil.virtual_memory().total,'hardware':'CPU only',
          'xgboost_build_info':xgb.build_info(),
          'packages':{'xgboost':xgb.__version__,'numpy':np.__version__,'pandas':pd.__version__,'psutil':psutil.__version__},
          'workers':settings['workers'],'threads_per_fit':settings['threads_per_fit'],
          'feature_names':feature_names(settings['features']),
          'training_examples_per_series':settings['history_days']-lookback(settings['features']),
          'training_windows':{o:{'start':str((pd.Timestamp(o)-pd.Timedelta(days=settings['history_days']-1)).date()),
                                 'first_label':str((pd.Timestamp(o)-pd.Timedelta(days=settings['history_days']-1-lookback(settings['features']))).date()),
                                 'end':o} for o in origins},
          'resource_granularity':'one pooled candidate/origin fit across all included series',
          'memory_measurement':'10ms sampled fresh-worker RSS, includes runtime overhead; brief peaks may be missed',
          'setup_download_seconds':None,'gpu_memory_bytes':None,'final_test_evaluated':False}
    (out/'run.json').write_text(json.dumps(info,indent=2)+'\n')
    print(f'{run_id}: {len(tasks)} pooled validation jobs; {len(ids)} series; {info["training_examples_per_series"]*len(ids)} training rows per job',flush=True)
    results=[]
    with ProcessPoolExecutor(max_workers=settings['workers'],max_tasks_per_child=1,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(run_pooled_task,task) for task in tasks]
        with (out/'task_log.jsonl').open('w') as log:
            for future in as_completed(futures):
                result=future.result();results.append(result)
                log.write(json.dumps({k:v for k,v in result.items() if k!='raw'})+'\n');log.flush()
                print(f'{len(results)}/{len(tasks)} {result["candidate"]} {result["forecast_origin"]}: {result["status"]}, {result["elapsed_seconds"]:.2f}s',flush=True)
    forecasts=[];cases=[];metrics=[];gains=[]
    for result in sorted(results,key=lambda r:(r['candidate'],r['forecast_origin'])):
        origin=result['forecast_origin']
        raw=np.asarray(result['raw']) if result['raw'] is not None else np.full((len(ids),max(horizons)),np.nan)
        prediction=np.maximum(raw,0)
        for feature,gain in result['feature_gain'].items():
            gains.append({'candidate':result['candidate'],'forecast_origin':origin,'feature':feature,'gain':gain})
        for index,sid in enumerate(ids):
            actual=actuals[(sid,origin)];train=histories[origin][index]
            common={'run_id':run_id,'model':'XGBoost','model_version':xgb.__version__,'candidate':result['candidate'],
                    'scenario':'univariate','history_window':settings['history_days'],'series_id':sid,
                    'forecast_origin':origin,'status':result['status']}
            scale,reason=rmsse_scale(train)
            for horizon in horizons:
                fields={**common,'horizon':horizon}
                values=score(train,actual[:horizon],prediction[index,:horizon]) if result['status']=='ok' else {m:(np.nan,result['error']) for m in ['MAE','WAPE','RMSSE']}
                for metric,(value,why) in values.items():
                    metrics.append({**fields,'metric':metric,'value':value,'valid':bool(np.isfinite(value)),
                                    'reason':why,'n_observations':horizon})
                cases.append({**fields,**{m:v[0] for m,v in values.items()},'scale_valid':bool(np.isfinite(scale)),
                              'scale_reason':reason,'actual_sum':float(actual[:horizon].sum()),
                              'absolute_error_sum':float(np.abs(actual[:horizon]-prediction[index,:horizon]).sum())})
                for lead in range(horizon):
                    forecasts.append({**fields,'lead':lead+1,'date':str(dates[origin][lead].date()),'y_true':actual[lead],
                                      'y_pred_raw':raw[index,lead],'y_pred':prediction[index,lead]})
    f=pd.DataFrame(forecasts);c=pd.DataFrame(cases)
    resource=pd.DataFrame([{k:v for k,v in r.items() if k not in ['raw','feature_gain']} for r in results])
    assert not f.duplicated(['candidate','series_id','forecast_origin','horizon','lead']).any()
    assert f.date.max()<=str((pd.Timestamp(protocol['validation_origins'][-1])+pd.Timedelta(days=max(horizons))).date())
    f.to_csv(forecasts_dir/'candidate_forecasts.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    c.to_csv(out/'case_scores.csv',index=False);pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    resource.to_csv(out/'resources.csv',index=False)
    pd.DataFrame(gains,columns=['candidate','forecast_origin','feature','gain']).to_csv(out/'feature_gain.csv',index=False)
    summary,resource_summary=summarize(c,resource)
    resource_summary=resource_summary.rename(columns={'scheduled_cases':'scheduled_pooled_jobs','failed_cases':'failed_pooled_jobs'})
    resource_summary['affected_series_failures']=resource_summary.failed_pooled_jobs*len(ids)
    summary.to_csv(out/'accuracy_by_horizon.csv',index=False)
    resource_summary.to_csv(out/'resource_summary.csv',index=False)
    ranking,selections=select_candidates(c,settings['candidates'],horizons)
    ranking.to_csv(out/'candidate_selection.csv',index=False)
    chosen=selections['XGBoost'];chosen_f=f.loc[f.candidate.eq(chosen)]
    chosen_f.to_csv(forecasts_dir/'forecasts.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(metrics).loc[lambda x:x.candidate.eq(chosen)].to_csv(out/'selected_metrics.csv',index=False)
    reliability,_=consistency(c,selections,len(origins));reliability.to_csv(out/'reliability.csv',index=False)
    c.groupby(['candidate','forecast_origin','horizon'])[['MAE','RMSSE']].agg(['median',lambda x:x.quantile(.9),lambda x:x.quantile(.95)]).to_csv(out/'origin_distributions.csv')
    selection={'run_id':run_id,'status':'smoke_only_not_selected' if smoke else 'validation_selected',
               'scenario':'univariate','history_days':settings['history_days'],'selected_candidates':selections,
               'candidate_specs':{'XGBoost':next((s for s in settings['candidates'] if s['id']==chosen),None)},
               'selection_config_sha256':sha(config_path),'protocol_sha256':sha(protocol_path),'test_evaluated':False}
    (out/'selected_models.json').write_text(json.dumps(selection,indent=2)+'\n')
    info.update(state='complete',elapsed_wall_seconds=time.perf_counter()-start,
                failed_pooled_jobs=int(resource.status.ne('ok').sum()),
                affected_series_failures=int(resource.status.ne('ok').sum())*len(ids),
                candidate_forecast_rows=len(f),selected_forecast_rows=len(chosen_f))
    (out/'run.json').write_text(json.dumps(info,indent=2)+'\n')
    print(ranking.to_string(index=False),flush=True);print('Results:',out,flush=True)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();run(smoke=args.smoke)
