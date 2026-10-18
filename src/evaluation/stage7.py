"""Audit and re-score all completed model forecasts with one validation protocol."""
import argparse
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.evaluation.stage4 import ROOT, sha, consistency
from src.evaluation.report_stage5 import verify_comparable, matched_cases, markdown
from src.evaluation.metrics import score, rmsse_scale, pooled_wape
from src.evaluation.windows import validation_data, make_window


JOB = ['candidate','series_id','forecast_origin']
CASE = JOB + ['horizon']


def audit_forecasts(frame, ids, origins, candidates, horizons=(7,14,28)):
    """Require the entire schedule, aligned dates, explicit failures and shared prefixes."""
    if frame.duplicated(CASE+['lead']).any():
        raise ValueError('Duplicate forecast row')
    if set(frame.series_id) != set(ids) or set(frame.forecast_origin) != set(origins):
        raise ValueError('Series/origin mismatch (test data is forbidden)')
    if set(frame.candidate) != set(candidates) or set(frame.horizon) != set(horizons):
        raise ValueError('Model/horizon mismatch')
    if not frame.scenario.eq('univariate').all() or not frame.history_window.eq(365).all():
        raise ValueError('Information condition mismatch')
    expected = pd.MultiIndex.from_product([candidates,ids,origins,horizons],names=CASE)
    counts = frame.groupby(CASE).size().reindex(expected)
    if counts.isna().any() or not np.array_equal(counts.to_numpy(),expected.get_level_values('horizon')):
        raise ValueError('Missing scheduled case or incorrect number of leads')
    if not ((frame.lead >= 1) & (frame.lead <= frame.horizon) & frame.lead.eq(frame.lead.astype(int))).all():
        raise ValueError('Invalid lead')
    expected_dates = pd.to_datetime(frame.forecast_origin) + pd.to_timedelta(frame.lead,unit='D')
    if not pd.to_datetime(frame.date).eq(expected_dates).all():
        raise ValueError('Forecast date is not origin plus lead')
    if frame.groupby(JOB).status.nunique().ne(1).any():
        raise ValueError('Inconsistent status across horizons')
    if not set(frame.status).issubset({'ok','failed','timeout'}):
        raise ValueError('Unknown forecast status')
    same = frame.groupby(JOB+['lead'])[['y_true','y_pred_raw','y_pred']].nunique(dropna=False)
    if same.gt(1).any().any():
        raise ValueError('Overlapping forecast prefixes differ')
    good = frame.status.eq('ok')
    if not np.isfinite(frame.loc[good,['y_true','y_pred_raw','y_pred']]).all().all():
        raise ValueError('Successful forecast has nonfinite values')
    if not np.allclose(frame.loc[good,'y_pred'],np.maximum(frame.loc[good,'y_pred_raw'],0),rtol=1e-12,atol=1e-12):
        raise ValueError('Clipping does not match the shared protocol')
    if frame.loc[~good,['y_pred_raw','y_pred']].notna().any().any():
        raise ValueError('Failed forecasts must not contain fallback predictions')


def bootstrap_comparison(cases, settings):
    """Paired resampling of whole series, preserving every origin and horizon."""
    models = sorted(cases.model.unique()); horizons = sorted(cases.horizon.unique())
    origins = sorted(cases.forecast_origin.unique())
    scheduled = cases.series_id.nunique()
    valid = cases.status.eq('ok') & cases[['MAE','RMSSE']].notna().all(axis=1)
    counts = cases.loc[valid].groupby('series_id').size()
    ids = sorted(counts[counts.eq(len(models)*len(horizons)*len(origins))].index)
    if not ids: raise ValueError('No complete common series for uncertainty analysis')
    common = cases.loc[cases.series_id.isin(ids)]
    rng = np.random.default_rng(settings['seed']); n = len(ids); repeats = settings['replicates']
    draws = rng.integers(n,size=(repeats,n))
    weights = np.zeros((repeats,n),dtype=float)
    np.add.at(weights,(np.repeat(np.arange(repeats),n),draws.ravel()),1)
    lower = (1-settings['confidence_level'])/2
    results=[]
    for metric in ['MAE','RMSSE','WAPE']:
        values=[]; samples=[]
        for model in models:
            group=common.loc[common.model.eq(model)]
            if metric=='WAPE':
                sums=group.groupby(['series_id','horizon'])[['absolute_error_sum','actual_sum']].sum()
                num=sums.absolute_error_sum.unstack('horizon').reindex(index=ids,columns=horizons).to_numpy()
                den=sums.actual_sum.unstack('horizon').reindex(index=ids,columns=horizons).to_numpy()
                observed=(num.sum(axis=0)/den.sum(axis=0)).mean()
                sampled=((weights @ num)/(weights @ den)).mean(axis=1)
            else:
                matrix=group.groupby(['series_id','horizon'])[metric].mean().unstack('horizon').reindex(index=ids,columns=horizons).to_numpy()
                observed=matrix.mean(); sampled=(weights @ matrix/n).mean(axis=1)
            values.append(observed); samples.append(sampled)
        for a,b in combinations(range(len(models)),2):
            delta=samples[a]-samples[b]
            lo,hi=np.quantile(delta,[lower,1-lower])
            results.append({'model_a':models[a],'model_b':models[b],'metric':metric,
                'difference_a_minus_b':values[a]-values[b], 'ci_lower':lo,'ci_upper':hi,
                'common_complete_series':n,'excluded_series':scheduled-n,'replicates':repeats,
                'horizon_weight':'1/3 each','interpretation':'negative favors model_a; descriptive validation interval'})
    return pd.DataFrame(results)


def run(root=ROOT):
    config_path=root/'configs/stage7.json'; protocol_path=root/'configs/evaluation.json'
    settings=json.loads(config_path.read_text()); protocol=json.loads(protocol_path.read_text())
    if settings['split']!='validation' or settings['history_days']!=365 or settings['scenario']!='univariate':
        raise ValueError('Stage 7 currently permits the locked primary validation condition only')
    if protocol['horizons_days']!=[7,14,28] or protocol['primary_history_days']!=365:
        raise ValueError('Locked protocol changed')
    if any(not np.isclose(w,1/3) for w in protocol['validation_model_selection']['horizon_weights'].values()):
        raise ValueError('Equal horizon weights are required')
    manifest=pd.read_csv(root/protocol['pilot_manifest']);ids=sorted(manifest.id)
    origins=protocol['validation_origins'];frames=[];cached=[];resources=[];choices={};provenance={}
    reference=None
    for stage,source in settings['source_runs'].items():
        folder=root/'results/metrics'/source
        info=json.loads((folder/'run.json').read_text())
        if info['state']!='complete' or info['smoke'] or info['split']!='validation' or info['actual_origins']!=origins:
            raise ValueError('Incomplete, smoke, or non-validation source run')
        if info['protocol']!=protocol: raise ValueError('Source protocol differs from current locked protocol')
        if reference: verify_comparable(reference,info)
        reference=info
        for path in [protocol['pilot_manifest'],'data/processed/pilot_daily.csv.gz','src/evaluation/metrics.py','src/evaluation/windows.py',f'configs/stage{stage}.json']:
            if sha(root/path)!=info['files_sha256'][path]:raise ValueError(f'Source input changed: {path}')
        selection=json.loads((folder/'selected_models.json').read_text())['selected_candidates']
        if any(not candidate for candidate in selection.values()):raise ValueError('Unselected model')
        choices.update(selection)
        path=root/'results/forecasts'/source/'forecasts.csv.gz'
        frame=pd.read_csv(path,dtype={'model_version':str},low_memory=False)
        if set(frame.candidate)!=set(selection.values()):raise ValueError('Forecasts disagree with selected candidates')
        for model,candidate in selection.items():
            if not frame.loc[frame.candidate.eq(candidate),'model'].eq(model).all():raise ValueError('Candidate/model mapping mismatch')
        audit_forecasts(frame,ids,origins,list(selection.values()))
        if stage!='6':
            frame['point_estimator']='local_point_forecast';frame['mean_comparison_eligible']=True
            frame['model_family']='locally_trained_machine_learning' if stage=='5' else 'locally_fitted_statistical'
            frame.loc[frame.model.isin(['last_value','weekly_seasonal_naive']),'model_family']='naive_reference_baselines'
            frame['pretraining_audit_status']='not_applicable'
        frames.append(frame)
        scores=pd.read_csv(folder/'case_scores.csv');cached.append(scores.loc[scores.candidate.isin(selection.values())])
        logs=pd.read_csv(folder/'resources.csv');logs=logs.loc[logs.candidate.isin(selection.values())].copy()
        logs['source_run']=source;logs['measurement_scope']='per_series_fit' if stage=='4' else 'pooled_origin' if stage=='5' else 'batched_origin_shared_checkpoint'
        resources.append(logs)
        for item in [path,folder/'run.json',folder/'selected_models.json',folder/'case_scores.csv',folder/'resources.csv']:
            provenance[str(item.relative_to(root))]=sha(item)
    expected={m for family in protocol['model_families'].values() if isinstance(family,dict) for m in family.get('models',[])}
    if set(choices)!=expected:raise ValueError('All eight locked models are required')
    forecasts=pd.concat(frames,ignore_index=True)
    audit_forecasts(forecasts,ids,origins,list(choices.values()))
    pilot=validation_data(pd.read_csv(root/'data/processed/pilot_daily.csv.gz',usecols=['series_id','date','units_sold'],parse_dates=['date']),protocol)
    series={sid:g.set_index('date').units_sold.sort_index() for sid,g in pilot.groupby('series_id')}
    windows={(sid,o):make_window(series[sid],o,365,28) for sid in ids for o in origins}
    cases=[];metric_rows=[]
    print('Auditing and rescoring 4,800 model/series/origin jobs...',flush=True)
    for (candidate,sid,origin,h),group in forecasts.groupby(CASE,sort=True):
        group=group.sort_values('lead');row=group.iloc[0];train,actual,_=windows[sid,origin]
        if not np.array_equal(actual[:h],group.y_true.to_numpy()):raise ValueError('Saved targets differ from original validation labels')
        good=row.status=='ok'
        values=score(train,actual[:h],group.y_pred.to_numpy()) if good else {m:(np.nan,'source_forecast_failed_no_fallback') for m in ['MAE','WAPE','RMSSE']}
        fields={k:row[k] for k in ['model','model_version','candidate','model_family','point_estimator','mean_comparison_eligible','pretraining_audit_status','scenario','history_window','series_id','forecast_origin','horizon','status']}
        scale,reason=rmsse_scale(train)
        cases.append({**fields,**{m:v[0] for m,v in values.items()},'scale_valid':bool(np.isfinite(scale)),'scale_reason':reason,
                      'actual_sum':actual[:h].sum(),'absolute_error_sum':np.abs(actual[:h]-group.y_pred.to_numpy()).sum()})
        for metric,(value,why) in values.items():
            metric_rows.append({**fields,'metric':metric,'value':value,'valid':bool(np.isfinite(value)),'reason':why,'n_observations':h})
    cases=pd.DataFrame(cases)
    old=pd.concat(cached).set_index(CASE).sort_index();new=cases.set_index(CASE).sort_index()
    if not old.index.equals(new.index):raise ValueError('Saved score grid mismatch')
    for column in ['MAE','WAPE','RMSSE','actual_sum','absolute_error_sum']:
        if not np.allclose(old[column],new[column],rtol=1e-10,atol=1e-10,equal_nan=True):raise ValueError(f'Recalculated scores differ: {column}')
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-stage7-validation'
    out=root/'results/metrics'/run_id;out.mkdir(parents=True,exist_ok=False)
    forecast_dir=root/'results/forecasts'/run_id;forecast_dir.mkdir(parents=True,exist_ok=False)
    forecasts=forecasts.rename(columns={'run_id':'source_run_id'});forecasts.insert(0,'run_id',run_id)
    forecasts.to_csv(forecast_dir/'forecasts.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    cases.insert(0,'run_id',run_id);cases.to_csv(out/'case_scores.csv',index=False)
    pd.DataFrame(metric_rows).assign(run_id=run_id).to_csv(out/'metrics.csv',index=False)
    resource_frame=pd.concat(resources,ignore_index=True);resource_frame.to_csv(out/'source_resources.csv',index=False)
    jobs=cases.drop_duplicates(JOB)
    coverage=jobs.groupby(['model','candidate']).agg(scheduled_jobs=('status','size'),failed_jobs=('status',lambda s:int(s.ne('ok').sum()))).reset_index()
    coverage.to_csv(out/'coverage.csv',index=False)
    cases.groupby(['model','forecast_origin','horizon'])[['MAE','RMSSE']].agg(['median',lambda x:x.quantile(.9),lambda x:x.quantile(.95)]).to_csv(out/'origin_distributions.csv')
    tables={}
    for track in settings['comparison_tracks']:
        frame=cases if track=='native_estimators' else cases.loc[cases.mean_comparison_eligible]
        selected={m:c for m,c in choices.items() if c in set(frame.candidate)}
        matched=matched_cases(frame,list(selected.values()));rows=[]
        for (model,estimator,h),group in matched.groupby(['model','point_estimator','horizon']):
            rows.append({'model':model,'point_estimator':estimator,'horizon':h,'common_cases':len(group),
                         'MAE':group.MAE.mean(),'RMSSE':group.RMSSE.mean(),'WAPE':pooled_wape(group.absolute_error_sum,group.actual_sum)})
        table=pd.DataFrame(rows);table.to_csv(out/f'accuracy_{track}.csv',index=False)
        tables[track]=table.groupby(['model','point_estimator'])[['MAE','RMSSE','WAPE']].mean().reset_index()
        tables[track].to_csv(out/f'summary_{track}.csv',index=False)
        reliability,wins=consistency(frame,selected,len(origins));reliability.to_csv(out/f'reliability_{track}.csv',index=False);wins.to_csv(out/f'win_rates_{track}.csv',index=False)
        bootstrap_comparison(frame,settings['bootstrap']).to_csv(out/f'paired_intervals_{track}.csv',index=False)
    columns=['id','demand_type','volume_band','state_id','store_id','cat_id']
    grouped=matched_cases(cases,list(choices.values())).merge(manifest[columns].rename(columns={'id':'series_id'}),on='series_id',validate='many_to_one')
    segments=[]
    for dimension in columns[1:]:
        for (model,h,label),group in grouped.groupby(['model','horizon',dimension]):
            segments.append({'dimension':dimension,'segment':label,'model':model,'horizon':h,'n_cases':len(group),'n_series':group.series_id.nunique(),
                             'MAE':group.MAE.mean(),'RMSSE':group.RMSSE.mean(),'WAPE':pooled_wape(group.absolute_error_sum,group.actual_sum)})
    pd.DataFrame(segments).to_csv(out/'segments_native_estimators.csv',index=False)
    summary=tables['native_estimators'].copy()
    for column in ['MAE','RMSSE','WAPE']:summary[column]=summary[column].map(lambda x:f'{x:.4f}')
    text=f'''# Stage 7 — unified validation evaluation

All eight methods are evaluated under protocol v1.0: 200 pilot series, the same three validation origins, 365 days of sales history, and 7/14/28-day horizons with equal 1/3 weights. **Final test periods remain untouched.**

Run: `{run_id}`. This execution reuses completed model predictions after checking their provenance, complete schedule, daily targets, dates, nonnegative clipping and identical horizon prefixes. Every score was recalculated from the pilot and checked against its source. Models were not retrained, retuned or rerun; original measured resource logs are retained.

## Shared accuracy

{markdown(summary)}

Lower errors are better. There are 599 common successful series/origin cases per horizon out of 600. The table uses this common support for every metric; WAPE pools absolute errors and sales within a horizon, then averages the three horizon values equally. Chronos-2 supplies a median; the others use the documented point forecasts. This is the **native-estimator** comparison; `summary_mean_only.csv` excludes Chronos-2. These are selected-model validation scores, not final test performance.

## Failures and uncertainty

{markdown(coverage[['model','scheduled_jobs','failed_jobs']])}

The single earlier ARIMA failure is preserved without fallback. Failure counts use 4,800 scheduled model/series/origin jobs, not duplicated horizon rows. Reliability summaries require all three origins per series and include median, p90, p95, spread, maximum and exclusions. Win rates use common complete series and split ties.

`paired_intervals_*.csv` reports all prespecified model pairs for MAE, RMSSE and WAPE using 2,000 paired whole-series bootstrap draws (seed 42), retaining all origins and horizons. Each comparison track uses its complete common series, so the intervals can have stricter support than the accuracy table. Differences are model A minus B; negative favors A. These are descriptive 95% percentile intervals, with no correction for multiple comparisons. Shared products/stores introduce dependence, and validation selection introduces optimism; these are not population-wide guarantees.

## Reproducibility and limitations

- `run.json`: source artifact hashes, evaluation settings, and successful audit checks.
- `case_scores.csv`, `metrics.csv`: newly calculated scores for all scheduled cases.
- `source_resources.csv`: original timings and sampled memory, labeled by per-series, pooled, or batched scope. No fresh runtime benchmark was performed; do not sum shared peak memory or interpret worker-time sums as wall time.
- `segments_native_estimators.csv`: fixed historical demand/volume groups, states, stores and categories, with support counts.
- Saved combined predictions: `results/forecasts/{run_id}/forecasts.csv.gz`.

Moirai's disclosed family pretraining corpus includes M5; disjoint pretraining is not certified for these checkpoints. See `docs/pretraining_audit.md`. Chronos-2's median and Moirai's noncommercial checkpoint restriction remain explicitly documented. This pilot excludes cold starts.

This completes the unified **365-day univariate validation evaluation**. Shorter-history selection, the separate business-feature experiment, freezing all remaining choices, and final-test evaluation remain pending under the locked protocol. This report does not authorize or execute test access.
'''
    (out/'report.md').write_text(text);(root/'docs/stage7_results.md').write_text(text)
    info={'run_id':run_id,'state':'complete','split':'validation','settings':settings,'protocol':protocol,
          'execution':'reuse_verified_predictions_recalculate_all_scores','selected_candidates':choices,
          'source_artifacts_sha256':provenance,'files_sha256':{str(p.relative_to(root)):sha(p) for p in [config_path,protocol_path,Path(__file__),root/'src/evaluation/metrics.py',root/protocol['pilot_manifest'],root/'data/processed/pilot_daily.csv.gz']},
          'scheduled_jobs':len(jobs),'failed_jobs':int(jobs.status.ne('ok').sum()),'forecast_rows':len(forecasts),'case_score_rows':len(cases),
          'checks':['full_schedule','source_data_hashes','protocol_identity','validation_only','target_values','date_alignment','shared_prefixes','clipping','no_failure_fallback','rescored_metrics_match'],
          'final_test_evaluated':False}
    (out/'run.json').write_text(json.dumps(info,indent=2)+'\n')
    print(summary.to_string(index=False));print('Results:',out,flush=True)
    return out


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args();run()
