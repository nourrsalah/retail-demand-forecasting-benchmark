"""Sequential frozen-checkpoint validation; final-test execution is not exposed."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import pandas as pd
import psutil
from src.evaluation.windows import make_window, validation_data
from src.evaluation.metrics import score, rmsse_scale, pooled_wape, equal_horizon_score
from src.evaluation.stage4 import ROOT, sha, consistency


def check_settings(settings, protocol):
    if settings['split'] != 'validation' or settings['scenario'] != 'univariate':
        raise ValueError('Only univariate validation is allowed')
    if settings['history_days'] != protocol['primary_history_days'] or settings['history_days'] != 365:
        raise ValueError('Locked history mismatch')
    if protocol['horizons_days'] != [7,14,28]:
        raise ValueError('Locked horizon mismatch')
    if any(not np.isclose(protocol['validation_model_selection']['horizon_weights'][str(h)], 1/3)
           for h in protocol['horizons_days']):
        raise ValueError('Equal horizon weights required')
    if settings['device'] != 'cpu' or settings['workers'] != 1:
        raise ValueError('This implementation uses one CPU worker at a time')


def verify_checkpoint(root, spec, lock):
    item = lock['models'][spec['id']]
    if item['repo_id'] != spec['repo_id'] or len(item['revision']) != 40:
        raise ValueError('Checkpoint identity mismatch')
    path = root/item['snapshot_path']
    for name, metadata in item['files'].items():
        if sha(path/name) != metadata['sha256']:
            raise ValueError(f'Checkpoint file changed: {spec["id"]}/{name}')
    return path


def run(root=ROOT, smoke=False, model=None):
    started = time.perf_counter()
    protocol = json.loads((root/'configs/evaluation.json').read_text())
    settings = json.loads((root/'configs/stage6.json').read_text()); check_settings(settings, protocol)
    lock = json.loads((root/settings['checkpoint_lock']).read_text())
    specs = [s for s in settings['models'] if model is None or s['id'] == model]
    if not specs: raise ValueError('Unknown model')
    checkpoints = {s['id']:verify_checkpoint(root, s, lock) for s in specs}
    manifest = pd.read_csv(root/protocol['pilot_manifest'])
    if smoke: manifest = manifest.sort_values('id').groupby('demand_type', sort=True).head(1)
    ids = sorted(manifest.id.tolist())
    origins = protocol['validation_origins'][:1] if smoke else protocol['validation_origins']
    horizons = protocol['horizons_days']
    data = pd.read_csv(root/'data/processed/pilot_daily.csv.gz',
                       usecols=['series_id','date','units_sold'], parse_dates=['date'])
    data = validation_data(data, protocol)
    series = {sid:g.set_index('date').units_sold.sort_index() for sid,g in data.groupby('series_id') if sid in ids}
    histories, actuals, dates = [], {}, {}
    for origin in origins:
        history = []
        for sid in ids:
            train, actual, targets = make_window(series[sid], origin, settings['history_days'], max(horizons))
            history.append(train); actuals[sid,origin] = actual; dates[origin] = targets
        histories.append(np.stack(history))
    histories = np.stack(histories)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + ('-stage6-smoke' if smoke else '-stage6-validation')
    out = root/'results/metrics'/run_id; out.mkdir(parents=True, exist_ok=False)
    forecast_dir = root/'results/forecasts'/run_id; forecast_dir.mkdir(parents=True, exist_ok=False)
    work = root/'.cache/stage6'/run_id; work.mkdir(parents=True, exist_ok=False)
    history_path = work/'history.npy'; np.save(history_path, histories, allow_pickle=False)
    paths = [root/'configs/evaluation.json', root/'configs/stage6.json', root/settings['checkpoint_lock'],
             root/'docs/evaluation_protocol.md', root/'docs/stage6.md', root/'docs/pretraining_audit.md',
             root/'requirements/stage6.lock.txt', root/'requirements/stage6.in', root/'src/utils/prepare_stage6.py', root/protocol['pilot_manifest'], root/'data/processed/pilot_daily.csv.gz']
    paths += sorted((root/'src/models').glob('*.py')) + sorted((root/'src/evaluation').glob('*.py'))
    info = {'run_id':run_id, 'state':'running', 'smoke':smoke, 'split':'validation', 'settings':settings,
            'protocol':protocol, 'checkpoints':lock, 'actual_origins':origins, 'series_ids':ids,
            'series_count':len(ids), 'models_run':[s['id'] for s in specs],
            'scheduled_series_cases':len(ids)*len(origins)*len(specs),
            'files_sha256':{str(p.relative_to(root)):sha(p) for p in paths},
            'git_revision':subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True).stdout.strip(),
            'platform':platform.platform(), 'python':platform.python_version(), 'machine':platform.machine(),
            'host_ram_bytes':psutil.virtual_memory().total, 'logical_cpus':os.cpu_count(),
            'hardware':'CPU only', 'precision':'float32', 'gpu_memory_bytes':None,
            'started_utc':datetime.now(timezone.utc).isoformat(), 'final_test_evaluated':False,
            'memory_measurement':'10ms sampled worker RSS; shared model peak repeated on origin rows; do not sum',
            'setup_dependency_seconds':None, 'fitting':'none; frozen zero-shot checkpoint',
            'training_windows':{o:{'start':str((pd.Timestamp(o)-pd.Timedelta(days=364)).date()),'end':o} for o in origins}}
    (out/'run.json').write_text(json.dumps(info,indent=2)+'\n')
    print(f'{run_id}: {len(ids)} series, {len(origins)} validation origins', flush=True)
    cases, forecasts, metrics, resources = [], [], [], []
    for spec in specs:
        key = spec['id']; worker_out = work/key; worker_out.mkdir()
        request = {'spec':spec, 'settings':settings, 'checkpoint':str(checkpoints[key]),
                   'history':str(history_path), 'origins':origins, 'output':str(worker_out)}
        request_path = worker_out/'request.json'; request_path.write_text(json.dumps(request,indent=2))
        print(f'Running {key} sequentially on CPU', flush=True)
        tic = time.perf_counter(); error = None
        with (out/f'{key}.log').open('w') as log:
            try:
                result = subprocess.run([str(root/settings['environment_python']), '-m',
                    'src.evaluation.foundation_worker', str(request_path)], cwd=root, stdout=log, stderr=subprocess.STDOUT,
                    timeout=settings['worker_timeout_seconds']*(len(origins)+1))
                if result.returncode: error = f'Worker exited {result.returncode}; see {key}.log'
            except subprocess.TimeoutExpired:
                error = 'Worker timeout; no fallback forecast'
        elapsed = time.perf_counter()-tic
        if error or not (worker_out/'resources.json').exists():
            logs = [{'candidate':key,'forecast_origin':o,'status':'failed','error':error or 'Worker output absent'} for o in origins]
        else:
            logs = json.loads((worker_out/'resources.json').read_text())
        environment = worker_out/'environment.json'
        if environment.exists(): (out/f'{key}_environment.json').write_text(environment.read_text())
        for i, origin in enumerate(origins):
            resource = logs[i]
            resource.update(worker_wall_seconds=elapsed if i == 0 else 0.0,
                            affected_series_failures=len(ids) if resource['status'] != 'ok' else 0)
            seconds=resource.get('inference_seconds')
            resource['first_inference_series_per_second']=len(ids)/seconds if seconds else None
            resources.append(resource)
            raw = np.load(worker_out/f'{i}.npy',allow_pickle=False) if resource['status']=='ok' else np.full((len(ids),28),np.nan)
            predicted = np.maximum(raw,0)
            for j,sid in enumerate(ids):
                train = histories[i,j]; actual = actuals[sid,origin]; scale,reason = rmsse_scale(train)
                common = {'run_id':run_id,'model':spec['model'],'model_version':lock['models'][key]['revision'],
                          'candidate':key,'model_family':'pretrained_time_series_foundation',
                          'point_estimator':spec['point_estimator'],'mean_comparison_eligible':spec['mean_comparison_eligible'],
                          'pretraining_audit_status':spec['pretraining_audit_status'],
                          'scenario':'univariate','history_window':365,'series_id':sid,'forecast_origin':origin,
                          'status':resource['status']}
                for h in horizons:
                    fields = {**common,'horizon':h}
                    scores = score(train,actual[:h],predicted[j,:h]) if resource['status']=='ok' else {m:(np.nan,resource['error']) for m in ['MAE','WAPE','RMSSE']}
                    cases.append({**fields,**{m:v[0] for m,v in scores.items()},'scale_valid':bool(np.isfinite(scale)),
                                  'scale_reason':reason,'actual_sum':float(actual[:h].sum()),
                                  'absolute_error_sum':float(np.abs(actual[:h]-predicted[j,:h]).sum())})
                    for m,(v,why) in scores.items():
                        metrics.append({**fields,'metric':m,'value':v,'valid':bool(np.isfinite(v)),
                                        'reason':why,'n_observations':h})
                    for lead in range(h):
                        forecasts.append({**fields,'lead':lead+1,'date':str(dates[origin][lead].date()),
                                          'y_true':actual[lead],'y_pred_raw':raw[j,lead],'y_pred':predicted[j,lead]})
        print(f'{key}: {[r["status"] for r in logs]}, {elapsed:.1f}s',flush=True)
    c = pd.DataFrame(cases); f = pd.DataFrame(forecasts); r = pd.DataFrame(resources)
    assert not f.duplicated(['candidate','series_id','forecast_origin','horizon','lead']).any()
    c.to_csv(out/'case_scores.csv',index=False); r.to_csv(out/'resources.csv',index=False)
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    f.to_csv(forecast_dir/'forecasts.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    rows = []
    for (key,h),g in c.groupby(['candidate','horizon']):
        good = g.loc[g.status.eq('ok')]
        rows.append({'candidate':key,'horizon':h,'scheduled_cases':len(g),'valid_cases':len(good),
                     'valid_rmsse_cases':int(good.RMSSE.notna().sum()),'MAE':good.MAE.mean(),
                     'RMSSE':good.RMSSE.mean(),'WAPE':pooled_wape(good.absolute_error_sum,good.actual_sum)})
    table = pd.DataFrame(rows); table.to_csv(out/'accuracy_by_horizon.csv',index=False)
    selection = []
    for spec in specs:
        g = table.loc[table.candidate.eq(spec['id'])].set_index('horizon')
        selection.append({'candidate':spec['id'],'point_estimator':spec['point_estimator'],
                          'mean_comparison_eligible':spec['mean_comparison_eligible'],
                          'selection_score':equal_horizon_score(g.RMSSE.to_dict()),
                          'mae_tie_break':equal_horizon_score(g.MAE.to_dict()),
                          'selection':'one prespecified checkpoint; no validation search'})
    pd.DataFrame(selection).to_csv(out/'candidate_selection.csv',index=False)
    choices = {s['model']:s['id'] for s in specs}
    reliability,_ = consistency(c,choices,len(origins)); reliability.to_csv(out/'reliability.csv',index=False)
    c.groupby(['candidate','forecast_origin','horizon'])[['MAE','RMSSE']].agg(['median',lambda x:x.quantile(.9),lambda x:x.quantile(.95)]).to_csv(out/'origin_distributions.csv')
    (out/'selected_models.json').write_text(json.dumps({'status':'smoke_only' if smoke else 'prespecified_checkpoints',
        'selected_candidates':choices,'test_evaluated':False},indent=2)+'\n')
    info.update(state='complete',elapsed_wall_seconds=time.perf_counter()-started,
                failed_series_cases=int(r.affected_series_failures.sum()),forecast_rows=len(f))
    (out/'run.json').write_text(json.dumps(info,indent=2)+'\n')
    print(table.to_string(index=False)); print('Results:',out,flush=True)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true'); parser.add_argument('--model',choices=['chronos2','timesfm25','moirai11small'])
    args=parser.parse_args();run(smoke=args.smoke,model=args.model)
