"""Validation selection, immutable freeze, then shared final-test execution."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import platform
import signal
import subprocess
import threading
import time
import numpy as np
import pandas as pd
import psutil
from src.evaluation.stage4 import ROOT, sha, run_task, select_candidates, _timeout
from src.evaluation.stage5 import run_pooled_task
from src.evaluation.stage6 import verify_checkpoint
from src.evaluation.windows import make_window, validation_data
from src.evaluation.metrics import score, rmsse_scale
from src.models.business_xgboost import business_features, fit_business, predict_business


def load(path):return json.loads(Path(path).read_text())
def save(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def key(history,scenario):return f'{scenario}_{history}'


def source_hashes():
    paths=sorted((ROOT/'src').rglob('*.py'))
    paths += [ROOT/p for p in ['configs/evaluation.json','configs/stage4.json','configs/stage5.json','configs/stage6.json',
        'configs/stage6_checkpoints.json','configs/stage7_experiments.json','requirements.txt','requirements/stage6.lock.txt',
        'docs/evaluation_protocol.md','docs/stage7_design.md','results/metrics/pilot_manifest.csv','data/processed/pilot_daily.csv.gz']]
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def verify_freeze(freeze):
    if freeze['status']!='frozen_before_final_test':raise ValueError('Final-test choices are not frozen')
    if source_hashes()!=freeze['source_sha256']:raise ValueError('Source/data/config changed after freeze; test execution refused')
    for path,digest in freeze['validation_artifacts_sha256'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Validation evidence changed after freeze')


def run_business_task(task):
    settings=task['settings'];spec=task['spec'];process=psutil.Process();samples=[process.memory_info().rss];stop=threading.Event()
    def monitor():
        while not stop.wait(.01):samples.append(process.memory_info().rss)
    thread=threading.Thread(target=monitor,daemon=True);thread.start();start=time.perf_counter()
    result={'candidate':spec['id'],'forecast_origin':task['forecast_origin'],'status':'ok','error':'','raw':None,'fit_seconds':np.nan,'inference_seconds':np.nan,'negative_raw_count':0}
    signal.signal(signal.SIGALRM,_timeout);signal.setitimer(signal.ITIMER_REAL,settings['fit_forecast_timeout_seconds'])
    try:
        tic=time.perf_counter();model=fit_business(task['history'],task['extras'],spec,settings);result['fit_seconds']=time.perf_counter()-tic
        tic=time.perf_counter();raw=predict_business(model,task['history'],task['future_extras'],settings);result['inference_seconds']=time.perf_counter()-tic
        times=[]
        for _ in range(settings['warm_inference_repeats']):
            tic=time.perf_counter();repeated=predict_business(model,task['history'],task['future_extras'],settings);times.append(time.perf_counter()-tic)
            if not np.allclose(raw,repeated,rtol=1e-12,atol=1e-12):raise ValueError('Warm forecasts changed')
        result.update(raw=raw.tolist(),negative_raw_count=int((raw<0).sum()),warm_inference_median_seconds=float(np.median(times)),warm_inference_range_seconds=float(np.ptp(times)))
    except Exception as exc:result.update(status='timeout' if isinstance(exc,TimeoutError) else 'failed',error=f'{type(exc).__name__}: {exc}')
    finally:signal.setitimer(signal.ITIMER_REAL,0);stop.set();thread.join()
    result.update(elapsed_seconds=time.perf_counter()-start,sampled_peak_rss_bytes=max(samples+[process.memory_info().rss]))
    return result


def run_condition(history,scenario,split,choices=None,freeze=None):
    if split=='test':
        if freeze is None:raise ValueError('No final-test access before freeze')
        verify_freeze(freeze)
        if choices!=freeze['selections'][key(history,scenario)]:raise ValueError('Test choices differ from freeze')
    elif split!='validation':raise ValueError('Unknown split')
    experiments=load(ROOT/'configs/stage7_experiments.json');protocol=load(ROOT/'configs/evaluation.json')
    if history not in experiments['histories'] or scenario not in ['univariate','business_features']:raise ValueError('Unsupported condition')
    if scenario=='business_features' and history!=365:raise ValueError('Business feature condition is 365 days only')
    configs={stage:load(ROOT/f'configs/stage{stage}.json') for stage in [4,5,6]}
    specs=(configs[4]['candidates']+configs[5]['candidates']+configs[6]['models']) if scenario=='univariate' else configs[5]['candidates']
    if choices:specs=[s for s in specs if s['id'] in choices.values()]
    manifest=pd.read_csv(ROOT/protocol['pilot_manifest']);ids=sorted(manifest.id);origins=protocol[f'{split}_origins'];horizons=protocol['horizons_days']
    pilot=pd.read_csv(ROOT/'data/processed/pilot_daily.csv.gz',parse_dates=['date'])
    if split=='validation':pilot=pilot.loc[pilot.date.le(pd.Timestamp(origins[-1])+pd.Timedelta(days=28))]
    grouped={sid:g.set_index('date',drop=False).sort_index() for sid,g in pilot.groupby('series_id')}
    histories=[];actuals={};dates={};historical_extras=[];future_extras=[]
    for origin in origins:
        rows=[];past_extra=[];future_extra=[]
        for sid in ids:
            group=grouped[sid];train,actual,target_dates=make_window(group.units_sold,origin,history,28)
            rows.append(train);actuals[sid,origin]=actual;dates[origin]=target_dates
            if scenario=='business_features':
                past=group.loc[pd.date_range(end=origin,periods=history)]
                # Explicit allowlist: future sales/prices never enter model requests.
                future=group.loc[target_dates,['date','event_name_1','event_name_2','snap_CA','snap_TX','snap_WI']]
                a,b=business_features(past,future,past.state_id.iloc[0]);past_extra.append(a);future_extra.append(b)
        histories.append(np.stack(rows))
        if past_extra:historical_extras.append(np.stack(past_extra));future_extras.append(np.stack(future_extra))
    histories=np.stack(histories)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+f'-stage7-{split}-{scenario}-{history}'
    out=ROOT/'results/metrics'/run_id;out.mkdir(parents=True)
    forecast_dir=ROOT/'results/forecasts'/run_id;forecast_dir.mkdir(parents=True)
    work=ROOT/'.cache/stage7'/run_id;work.mkdir(parents=True)
    info={'run_id':run_id,'state':'running','split':split,'scenario':scenario,'history_days':history,'origins':origins,'series_ids':ids,'specs':specs,
          'protocol':protocol,'source_sha256':source_hashes(),'platform':platform.platform(),'machine':platform.machine(),'RAM_bytes':psutil.virtual_memory().total,
          'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','statsmodels','xgboost','psutil']},
          'freeze_sha256':sha(ROOT/'configs/stage7_freeze.json') if split=='test' else None,
          'threads':{'statistical':1,'XGBoost':4,'foundation':4},'batch_size_foundation':4,'warm_repeats':3,'gpu_memory_bytes':None,
          'test_evaluated':split=='test','started_utc':datetime.now(timezone.utc).isoformat()}
    save(out/'run.json',info);start=time.perf_counter();results=[]
    print(f'{run_id}: {len(specs)} candidates, {len(ids)} series, {len(origins)} origins',flush=True)
    stats=[s for s in specs if s['id'] in {s['id'] for s in configs[4]['candidates']}]
    tasks=[{'spec':s,'candidate':s['id'],'series_id':sid,'forecast_origin':origin,'train':histories[i,j],'horizon':28,'settings':configs[4]}
           for i,origin in enumerate(origins) for j,sid in enumerate(ids) for s in stats]
    if tasks:
        with ProcessPoolExecutor(max_workers=experiments['workers_statistical'],mp_context=multiprocessing.get_context('spawn')) as pool:
            pending=[pool.submit(run_task,t) for t in tasks]
            for i,future in enumerate(as_completed(pending),1):
                result=future.result();result['scope']='per_series';results.append(result)
                if i%400==0:print(f'Statistical jobs: {i}/{len(tasks)}',flush=True)
    xgb=[s for s in specs if s['model']=='XGBoost']
    if xgb:
        with ProcessPoolExecutor(max_workers=1,max_tasks_per_child=1,mp_context=multiprocessing.get_context('spawn')) as pool:
            for i,origin in enumerate(origins):
                for spec in xgb:
                    task={'spec':spec,'history':histories[i],'forecast_origin':origin,'horizon':28,'settings':configs[5]}
                    if scenario=='business_features':task.update(extras=historical_extras[i],future_extras=future_extras[i])
                    result=pool.submit(run_business_task if scenario=='business_features' else run_pooled_task,task).result()
                    result['scope']='pooled';results.append(result);print(f'XGBoost {spec["id"]} {origin}: {result["status"]}',flush=True)
    foundation=[s for s in specs if s['id'] in {s['id'] for s in configs[6]['models']}]
    if foundation:
        lock=load(ROOT/configs[6]['checkpoint_lock']);np.save(work/'history.npy',histories,allow_pickle=False)
        for spec in foundation:
            checkpoint=verify_checkpoint(ROOT,spec,lock);folder=work/spec['id'];folder.mkdir()
            setting={**configs[6],'history_days':history}
            request={'spec':spec,'settings':setting,'checkpoint':str(checkpoint),'history':str(work/'history.npy'),'origins':origins,'output':str(folder)}
            save(folder/'request.json',request);print(f'{spec["model"]}: starting {len(origins)} origins',flush=True);tic=time.perf_counter();failure=None
            with (out/f'{spec["id"]}.log').open('w') as log:
                try:
                    process=subprocess.run([str(ROOT/setting['environment_python']),'-m','src.evaluation.foundation_worker',str(folder/'request.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=setting['worker_timeout_seconds']*(len(origins)+1))
                    if process.returncode:failure=f'Worker exit {process.returncode}'
                except subprocess.TimeoutExpired:failure='Worker timeout'
            records=load(folder/'resources.json') if not failure and (folder/'resources.json').exists() else [{'candidate':spec['id'],'forecast_origin':o,'status':'failed','error':failure or 'Missing worker output'} for o in origins]
            for i,result in enumerate(records):
                result['raw']=np.load(folder/f'{i}.npy').tolist() if result['status']=='ok' else None
                result.update(scope='batched',worker_wall_seconds=time.perf_counter()-tic if i==0 else 0);results.append(result)
            if (folder/'environment.json').exists():(out/f'{spec["id"]}_environment.json').write_text((folder/'environment.json').read_text())
            print(f'{spec["model"]}: {[r["status"] for r in records]}',flush=True)
    cases=[];forecasts=[];metrics=[];resource_rows=[];spec_map={s['id']:s for s in specs}
    for result in results:
        spec=spec_map[result['candidate']];origin=result['forecast_origin'];i=origins.index(origin)
        resource_rows.append({k:v for k,v in result.items() if k not in ['raw','feature_gain']})
        targets=[ids.index(result['series_id'])] if result['scope']=='per_series' else range(len(ids))
        raw=np.asarray(result['raw']) if result['raw'] is not None else None
        for j in targets:
            raw_row=raw if result['scope']=='per_series' else raw[j] if raw is not None else None
            if raw_row is None:raw_row=np.full(28,np.nan)
            prediction=np.maximum(raw_row,0);sid=ids[j];actual=actuals[sid,origin];train=histories[i,j]
            common={'run_id':run_id,'candidate':spec['id'],'model':spec['model'],'series_id':sid,'forecast_origin':origin,'scenario':scenario,'history_window':history,'status':result['status'],
                    'point_estimator':spec.get('point_estimator','local_point_forecast'),'mean_comparison_eligible':spec.get('mean_comparison_eligible',True),
                    'model_version':load(ROOT/configs[6]['checkpoint_lock'])['models'][spec['id']]['revision'] if spec in foundation else info['packages']['xgboost'] if spec['model']=='XGBoost' else info['packages']['statsmodels'] if spec['model'] in ['ARIMA','ETS'] else 'reference_rule_v1',
                    'model_family':spec.get('family','pretrained_time_series_foundation'),'pretraining_audit_status':spec.get('pretraining_audit_status','not_applicable')}
            scale,reason=rmsse_scale(train)
            for h in horizons:
                fields={**common,'horizon':h};values=score(train,actual[:h],prediction[:h]) if result['status']=='ok' else {m:(np.nan,result['error']) for m in ['MAE','WAPE','RMSSE']}
                cases.append({**fields,**{m:v[0] for m,v in values.items()},'scale_valid':bool(np.isfinite(scale)),'scale_reason':reason,'actual_sum':actual[:h].sum(),'absolute_error_sum':np.abs(actual[:h]-prediction[:h]).sum()})
                for metric,(value,why) in values.items():metrics.append({**fields,'metric':metric,'value':value,'reason':why,'valid':bool(np.isfinite(value)),'n_observations':h})
                for lead in range(h):forecasts.append({**fields,'lead':lead+1,'date':str(dates[origin][lead].date()),'y_true':actual[lead],'y_pred_raw':raw_row[lead],'y_pred':prediction[lead]})
    cases=pd.DataFrame(cases);forecast_frame=pd.DataFrame(forecasts)
    cases.to_csv(out/'case_scores.csv',index=False);pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False);pd.DataFrame(resource_rows).to_csv(out/'resources.csv',index=False)
    forecast_frame.to_csv(forecast_dir/'forecasts.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    if split=='validation':
        ranking,selected=select_candidates(cases,specs,horizons);ranking.to_csv(out/'candidate_selection.csv',index=False)
        if any(v is None for v in selected.values()):
            info.update(state='failed_validation_selection',error='At least one family has no eligible candidate')
            save(out/'run.json',info)
            raise RuntimeError('A family could not be selected; final test remains blocked')
    else:selected=choices
    save(out/'selected_models.json',{'selected_candidates':selected,'selection_data':'validation_only','test_tuning':False})
    info.update(state='complete',wall_seconds=time.perf_counter()-start,scheduled_jobs=len(cases.drop_duplicates(['candidate','series_id','forecast_origin'])),failed_jobs=len(cases.loc[cases.status.ne('ok')].drop_duplicates(['candidate','series_id','forecast_origin'])))
    save(out/'run.json',info);print('Completed:',out,flush=True);return out


def validate():
    index_path=ROOT/'results/metrics/stage7_validation_index.json'
    if (ROOT/'configs/stage7_freeze.json').exists():raise ValueError('Freeze already exists; do not retune after test')
    index=load(index_path) if index_path.exists() else {}
    for history,scenario in [(90,'univariate'),(180,'univariate'),(365,'business_features')]:
        name=key(history,scenario)
        if name in index:
            folder=ROOT/index[name];info=load(folder/'run.json')
            if info['state']!='complete' or info['source_sha256']!=source_hashes():raise ValueError('Existing validation source changed')
            print('Reusing completed condition:',name,flush=True);continue
        folder=run_condition(history,scenario,'validation');index[name]=str(folder.relative_to(ROOT));save(index_path,index)
    return index


def freeze_choices():
    path=ROOT/'configs/stage7_freeze.json'
    if path.exists():verify_freeze(load(path));print('Existing freeze verified');return
    index=load(ROOT/'results/metrics/stage7_validation_index.json');selections={};evidence={}
    for name in ['univariate_90','univariate_180','business_features_365']:
        folder=ROOT/index[name];info=load(folder/'run.json')
        if info['state']!='complete' or info['split']!='validation' or info['source_sha256']!=source_hashes():raise ValueError('Validation incomplete or source changed')
        selections[name]=load(folder/'selected_models.json')['selected_candidates']
        for file in ['run.json','selected_models.json','case_scores.csv','candidate_selection.csv']:
            p=folder/file;evidence[str(p.relative_to(ROOT))]=sha(p)
    primary={}
    for source in load(ROOT/'configs/stage7.json')['source_runs'].values():
        folder=ROOT/'results/metrics'/source
        primary.update(load(folder/'selected_models.json')['selected_candidates'])
        for file in ['run.json','selected_models.json','case_scores.csv']:
            p=folder/file;evidence[str(p.relative_to(ROOT))]=sha(p)
    selections['univariate_365']=primary
    snapshot={'status':'frozen_before_final_test','created_utc':datetime.now(timezone.utc).isoformat(),'selections':selections,
              'source_sha256':source_hashes(),'validation_artifacts_sha256':evidence,'experiment_design':load(ROOT/'configs/stage7_experiments.json'),
              'protocol':load(ROOT/'configs/evaluation.json'),'checkpoint_lock':load(ROOT/'configs/stage6_checkpoints.json')}
    save(path,snapshot);print('All choices frozen before test:',path,flush=True)


def final_test():
    freeze=load(ROOT/'configs/stage7_freeze.json');verify_freeze(freeze)
    path=ROOT/'results/metrics/stage7_test_index.json';index=load(path) if path.exists() else {}
    for history,scenario in [(365,'univariate'),(90,'univariate'),(180,'univariate'),(365,'business_features')]:
        name=key(history,scenario)
        if name in index:
            info=load(ROOT/index[name]/'run.json')
            if info['state']!='complete' or info['freeze_sha256']!=sha(ROOT/'configs/stage7_freeze.json'):raise ValueError('Existing test provenance mismatch')
            print('Preserving completed test condition:',name,flush=True);continue
        folder=run_condition(history,scenario,'test',freeze['selections'][name],freeze)
        index[name]=str(folder.relative_to(ROOT));save(path,index)
    return index


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['validate','freeze','test'])
    action=parser.parse_args().action
    {'validate':validate,'freeze':freeze_choices,'test':final_test}[action]()
