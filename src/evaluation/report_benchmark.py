"""Final frozen benchmark report; no test-dependent parameter selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.evaluation.benchmark import ROOT, load, save, verify_freeze
from src.evaluation.stage4 import consistency, sha
from src.evaluation.stage7 import audit_forecasts, bootstrap_comparison, CASE
from src.evaluation.report_stage5 import matched_cases, markdown
from src.evaluation.metrics import pooled_wape, score
from src.evaluation.windows import make_window


def tables(cases,folder,prefix,bootstrap_settings):
    choices=dict(cases[['model','candidate']].drop_duplicates().itertuples(index=False,name=None))
    common=matched_cases(cases,list(choices.values()));rows=[]
    for (model,h),g in common.groupby(['model','horizon']):
        rows.append({'model':model,'horizon':h,'common_cases':len(g),'MAE':g.MAE.mean(),'RMSSE':g.RMSSE.mean(),'WAPE':pooled_wape(g.absolute_error_sum,g.actual_sum)})
    table=pd.DataFrame(rows);table.to_csv(folder/f'{prefix}_by_horizon.csv',index=False)
    summary=table.groupby('model')[['MAE','RMSSE','WAPE']].mean().reset_index();summary.to_csv(folder/f'{prefix}_summary.csv',index=False)
    reliability,wins=consistency(cases,choices,cases.forecast_origin.nunique());reliability.to_csv(folder/f'{prefix}_reliability.csv',index=False);wins.to_csv(folder/f'{prefix}_win_rates.csv',index=False)
    bootstrap_comparison(cases,bootstrap_settings).to_csv(folder/f'{prefix}_paired_intervals.csv',index=False)
    return summary


def run():
    freeze=load(ROOT/'configs/stage7_freeze.json');verify_freeze(freeze)
    index=load(ROOT/'results/metrics/stage7_test_index.json')
    if set(index)!=set(freeze['selections']):raise ValueError('All frozen test conditions must be complete')
    out=ROOT/'results/metrics'/('stage7-final-'+sha(ROOT/'configs/stage7_freeze.json')[:12]);out.mkdir(exist_ok=True)
    protocol=freeze['protocol'];origins=protocol['test_origins'];manifest=pd.read_csv(ROOT/protocol['pilot_manifest']);ids=sorted(manifest.id)
    pilot=pd.read_csv(ROOT/'data/processed/pilot_daily.csv.gz',usecols=['series_id','date','units_sold'],parse_dates=['date'])
    series={sid:g.set_index('date').units_sold.sort_index() for sid,g in pilot.groupby('series_id')}
    frames={};resources=[];coverage=[];summaries=[];provenance={}
    bootstrap_settings={'seed':42,'replicates':2000,'confidence_level':.95}
    for condition,path in index.items():
        folder=ROOT/path;info=load(folder/'run.json');c=pd.read_csv(folder/'case_scores.csv')
        if info['state']!='complete' or info['split']!='test' or info['freeze_sha256']!=sha(ROOT/'configs/stage7_freeze.json'):raise ValueError('Test condition does not match freeze')
        selected=freeze['selections'][condition]
        if set(c.candidate)!=set(selected.values()):raise ValueError('Test contains unselected candidates')
        fpath=ROOT/'results/forecasts'/info['run_id']/'forecasts.csv.gz';f=pd.read_csv(fpath,low_memory=False)
        audit_forecasts(f.assign(history_window=365,scenario='univariate'),ids,origins,list(selected.values()))
        if not f.history_window.eq(info['history_days']).all() or not f.scenario.eq(info['scenario']).all():raise ValueError('Test condition mismatch')
        windows={(sid,o):make_window(series[sid],o,info['history_days'],28) for sid in ids for o in origins}
        checked=[]
        for (candidate,sid,origin,h),g in f.groupby(CASE):
            g=g.sort_values('lead');train,actual,_=windows[sid,origin]
            if not np.array_equal(actual[:h],g.y_true):raise ValueError('Final targets mismatch')
            values=score(train,actual[:h],g.y_pred.to_numpy()) if g.status.iloc[0]=='ok' else {m:(np.nan,'failed') for m in ['MAE','WAPE','RMSSE']}
            checked.append({'candidate':candidate,'series_id':sid,'forecast_origin':origin,'horizon':h,**{m:v[0] for m,v in values.items()}})
        checked=pd.DataFrame(checked).set_index(CASE).sort_index();saved=c.set_index(CASE).sort_index()
        if not checked.index.equals(saved.index) or not np.allclose(checked[['MAE','WAPE','RMSSE']],saved[['MAE','WAPE','RMSSE']],equal_nan=True,rtol=1e-10,atol=1e-10):raise ValueError('Final scores failed independent recalculation')
        frames[condition]=c
        jobs=c.drop_duplicates(['candidate','series_id','forecast_origin'])
        for model,g in jobs.groupby('model'):coverage.append({'condition':condition,'model':model,'scheduled_jobs':len(g),'failures':int(g.status.ne('ok').sum())})
        r=pd.read_csv(folder/'resources.csv');r['condition']=condition;r['source_run']=info['run_id'];resources.append(r)
        for path in [folder/'run.json',folder/'case_scores.csv',folder/'resources.csv',fpath]:provenance[str(path.relative_to(ROOT))]=sha(path)
        if info['scenario']=='univariate':
            for track,frame in [('native_estimators',c),('mean_only',c.loc[c.mean_comparison_eligible])]:
                summary=tables(frame,out,f'{condition}_{track}',bootstrap_settings);summary['history_days']=info['history_days'];summary['track']=track;summaries.append(summary)
        print('Audited final condition:',condition,flush=True)
    summary=pd.concat(summaries,ignore_index=True);summary.to_csv(out/'univariate_history_summary.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False);resource=pd.concat(resources,ignore_index=True);resource.to_csv(out/'resources.csv',index=False)
    resource_summary=[]
    for (condition,candidate),g in resource.groupby(['condition','candidate']):
        resource_summary.append({'condition':condition,'candidate':candidate,'scope':g.scope.iloc[0],
            'fit_seconds_sum':g.fit_seconds.sum(min_count=1),'first_inference_seconds_sum':g.inference_seconds.sum(min_count=1),
            'load_seconds_sum':g.load_seconds.sum(min_count=1) if 'load_seconds' in g else np.nan,
            'peak_sampled_worker_RSS_MiB':g.sampled_peak_rss_bytes.max()/1024**2,
            'warm_median_seconds_sum':g.warm_inference_median_seconds.sum(min_count=1)})
    pd.DataFrame(resource_summary).to_csv(out/'resource_summary.csv',index=False)
    base=frames['univariate_365'].loc[lambda x:x.model.eq('XGBoost')].copy()
    business=frames['business_features_365'].copy();business['model']='XGBoost + business features';business['candidate']='business_'+business.candidate
    feature_summary=tables(pd.concat([base,business],ignore_index=True),out,'business_features_within_XGBoost',bootstrap_settings)
    all_cases=pd.concat(frames.values(),ignore_index=True);all_cases.to_csv(out/'case_scores.csv',index=False)
    all_cases.groupby(['scenario','history_window','model','forecast_origin','horizon'])[['MAE','RMSSE']].agg(['median',lambda x:x.quantile(.9),lambda x:x.quantile(.95)]).to_csv(out/'origin_distributions.csv')
    segments=[]
    for condition,c in frames.items():
        selected=freeze['selections'][condition];matched=matched_cases(c,list(selected.values()))
        matched=matched.merge(manifest.rename(columns={'id':'series_id'})[['series_id','demand_type','volume_band','state_id','store_id','cat_id']],on='series_id',validate='many_to_one')
        for dimension in ['demand_type','volume_band','state_id','store_id','cat_id']:
            for (model,h,label),g in matched.groupby(['model','horizon',dimension]):
                segments.append({'condition':condition,'model':model,'horizon':h,'dimension':dimension,'segment':label,'n_cases':len(g),'n_series':g.series_id.nunique(),
                    'MAE':g.MAE.mean(),'RMSSE':g.RMSSE.mean(),'WAPE':pooled_wape(g.absolute_error_sum,g.actual_sum)})
    pd.DataFrame(segments).to_csv(out/'segments.csv',index=False)
    primary=summary.loc[summary.history_days.eq(365)&summary.track.eq('native_estimators'),['model','MAE','RMSSE','WAPE']].sort_values('RMSSE')
    def formatted(table):
        table=table.copy()
        for column in ['MAE','RMSSE','WAPE']:
            if column in table:table[column]=table[column].map(lambda x:f'{x:.4f}')
        return markdown(table)
    coverage=pd.DataFrame(coverage)
    text=f'''# Stage 7 — final shared-protocol evaluation

**Final test complete.** All model and experiment choices were frozen before any final-test forecasts. The freeze is `configs/stage7_freeze.json` ({freeze['created_utc']}); its source, configuration, checkpoint and validation evidence hashes still match.

The benchmark uses the original 200 pilot series, four final origins (2016-01-31, 2016-02-28, 2016-03-27, 2016-04-24), sales histories of 90/180/365 days, and 7/14/28-day forecasts weighted equally. Local models refit using only the available history at each origin; frozen foundation models receive exactly that observed history. The separate 365-day business-feature condition is a within-XGBoost comparison.

## Primary 365-day final results

{formatted(primary)}

Lower is better. These native-estimator results compare Chronos-2's median with the other documented point estimates. The separate mean-only tables exclude Chronos-2. Common successful RMSSE-eligible cases are used across models within each track; coverage and failures are preserved separately. WAPE pools errors and demand within each horizon, then averages horizon values equally.

## History sensitivity

{formatted(summary.loc[summary.track.eq('native_estimators'),['history_days','model','MAE','WAPE']].sort_values(['model','history_days']))}

MAE/WAPE compare identical targets across history lengths. RMSSE uses a different history-specific denominator, so use it to compare models **within** a history, not as a direct measure of improvement across histories. The 365-day validation experiment reused the already completed run; all final-test conditions are newly executed with their validation-selected settings.

## Separate business-feature experiment

{formatted(feature_summary)}

This compares XGBoost with and without known-calendar and historical-price information. Events/SNAP are assumed scheduled and known at the origin; future prices are carried forward from the last observed price. No realized future prices or target labels enter forecasting. Other covariate adapters were not implemented in this experiment; this is not a family-level information-parity ranking.

## Coverage and evidence

{markdown(coverage)}

Total: {coverage.scheduled_jobs.sum():,} scheduled model/series/origin jobs; {coverage.failures.sum()} failures, retained without fallbacks. Horizon rows do not inflate the failure denominator. Every saved final forecast was independently checked for complete scheduling, target/date alignment, clipping and identical overlapping prefixes; every score was recalculated and matched.

Saved in `results/metrics/{out.name}/`:

- Per-history, per-horizon mean-only and native-estimator accuracy.
- Across-series reliability (median/p90/p95, spread and worst cases) and shared-complete-series win rates.
- Paired 95% bootstrap intervals for all model pairs and MAE/RMSSE/WAPE: 2,000 whole-series draws, seed 42, preserving all origins and horizons. A negative A-minus-B difference favors A. Support is complete common series; intervals are descriptive, unadjusted for multiple comparisons, and shared store/product dependence limits population claims.
- `resource_summary.csv` and `resources.csv`: fitting/loading/first inference/warm timings and sampled RSS at the measured per-series, pooled or batched scope. Shared peak memory is not additive. Statistical workers are reused, so their RSS can retain allocations from earlier jobs. Times are host-specific; no profit or inventory savings are inferred.
- `segments.csv`, `origin_distributions.csv`, `case_scores.csv`, and `provenance.json`.

## Interpretation limits

Moirai's published family pretraining corpus includes M5; no clean M5-disjoint pretraining guarantee is established for these checkpoints. Final holdout discipline in this project does not undo possible external pretraining overlap. Chronos-2's median must not be described as a mean. Moirai's selected checkpoint is noncommercial. The diversity-balanced pilot excludes cold starts and does not represent every retail product.

No model choices were changed from final-test performance. Any new tuning, feature expansion, or model selection after this evaluation requires a fresh holdout or explicit exploratory labeling. This stage completes the planned shared-protocol runs; broader interpretation and business conclusions remain subsequent work.
'''
    (out/'report.md').write_text(text);(ROOT/'docs/stage7_final_results.md').write_text(text)
    save(out/'provenance.json',{'freeze_sha256':sha(ROOT/'configs/stage7_freeze.json'),'source_artifacts_sha256':provenance,'report_source_sha256':sha(Path(__file__)),'test_conditions':index,'checks_passed':True})
    print(primary.to_string(index=False));print('Final report:',ROOT/'docs/stage7_final_results.md',flush=True)
    return out


if __name__=='__main__':run()
