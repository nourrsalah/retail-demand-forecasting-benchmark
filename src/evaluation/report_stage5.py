"""Compare selected Stage 5 and frozen Stage 4 validation forecasts fairly."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from src.evaluation.metrics import pooled_wape
from src.evaluation.stage4 import ROOT, consistency, sha


def matched_cases(cases, expected_candidates):
    key=['series_id','forecast_origin','horizon']
    if cases.duplicated(['candidate']+key).any():
        raise ValueError('Duplicate comparison case')
    good=cases.loc[cases.status.eq('ok') & cases.RMSSE.notna() & cases.MAE.notna()]
    counts=good.groupby(key).size()
    keys=counts[counts.eq(len(expected_candidates))].index
    return good.set_index(key).loc[keys].reset_index() if len(keys) else good.iloc[:0]


def verify_comparable(old, new):
    for key in ['horizons_days','primary_history_days','validation_origins','pilot_manifest','validation_model_selection']:
        if old['protocol'][key]!=new['protocol'][key]:
            raise ValueError(f'Stage 4/5 protocol mismatch: {key}')
    for key in ['actual_origins','series_count']:
        if old[key]!=new[key]:raise ValueError(f'Stage 4/5 run mismatch: {key}')
    for path in [new['protocol']['pilot_manifest'],'data/processed/pilot_daily.csv.gz']:
        if old['files_sha256'][path]!=new['files_sha256'][path]:
            raise ValueError(f'Stage 4/5 data mismatch: {path}')
    if old['settings']['history_days']!=new['settings']['history_days'] or old['settings']['scenario']!=new['settings']['scenario']:
        raise ValueError('History or scenario mismatch')


def markdown(frame):
    return ('| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+
            '\n'.join('| '+' | '.join(str(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None)))


def write_report(run_dir):
    run_dir=Path(run_dir).resolve();info=json.loads((run_dir/'run.json').read_text())
    if info['state']!='complete' or info['smoke'] or info['split']!='validation':
        raise ValueError('Use a completed full validation run')
    old_dir=ROOT/'results/metrics'/info['settings']['stage4_reference_run']
    old_info=json.loads((old_dir/'run.json').read_text());verify_comparable(old_info,info)
    old_selected=json.loads((old_dir/'selected_models.json').read_text())['selected_candidates']
    selected=json.loads((run_dir/'selected_models.json').read_text())['selected_candidates']
    if not selected['XGBoost']:raise ValueError('No XGBoost candidate could be selected')
    all_selected={**old_selected,**selected}
    frames=[]
    for folder,choices in [(old_dir,old_selected),(run_dir,selected)]:
        c=pd.read_csv(folder/'case_scores.csv')
        frames.append(c.loc[c.candidate.isin([v for v in choices.values() if v])])
    combined=pd.concat(frames,ignore_index=True)
    # Assert target sums align before comparing independent saved experiments.
    if combined.groupby(['series_id','forecast_origin','horizon']).actual_sum.nunique().gt(1).any():
        raise ValueError('Scoring targets disagree across runs')
    matched=matched_cases(combined,list(all_selected.values()))
    if matched.empty:raise ValueError('No common comparison support')
    rows=[]
    for (model,candidate,horizon),g in matched.groupby(['model','candidate','horizon']):
        rows.append({'model':model,'candidate':candidate,'horizon':horizon,'common_cases':len(g),
                     'MAE':g.MAE.mean(),'RMSSE':g.RMSSE.mean(),
                     'WAPE':pooled_wape(g.absolute_error_sum,g.actual_sum)})
    table=pd.DataFrame(rows);table.to_csv(run_dir/'stage4_stage5_comparison.csv',index=False)
    reliability,wins=consistency(combined,all_selected,len(info['actual_origins']))
    reliability.to_csv(run_dir/'combined_reliability.csv',index=False);wins.to_csv(run_dir/'combined_win_rates.csv',index=False)
    # Labels are fixed by the Stage 2 historical sampling window, never today's results.
    manifest=pd.read_csv(ROOT/info['protocol']['pilot_manifest']).rename(columns={'id':'series_id'})
    segmented=matched.merge(manifest[['series_id','demand_type','volume_band','cat_id','store_id']],on='series_id',validate='many_to_one')
    segment_rows=[]
    for dimension in ['demand_type','volume_band','cat_id','store_id']:
        for (model,horizon,label),g in segmented.groupby(['model','horizon',dimension]):
            segment_rows.append({'model':model,'horizon':horizon,'dimension':dimension,'segment':label,
                                 'n_cases':len(g),'n_series':g.series_id.nunique(),'MAE':g.MAE.mean(),
                                 'RMSSE':g.RMSSE.mean(),'WAPE':pooled_wape(g.absolute_error_sum,g.actual_sum)})
    pd.DataFrame(segment_rows).to_csv(run_dir/'comparison_by_segment.csv',index=False)
    resource_rows=[]
    for folder,choices,scope in [(old_dir,old_selected,'per_series'),(run_dir,selected,'pooled')]:
        logs=pd.read_csv(folder/'resources.csv')
        for model,candidate in choices.items():
            g=logs.loc[logs.candidate.eq(candidate)]
            resource_rows.append({'model':model,'candidate':candidate,'job_scope':scope,'scheduled_jobs':len(g),
                                  'failed_jobs':int(g.status.ne('ok').sum()),'fit_seconds_sum':g.fit_seconds.sum(),
                                  'first_inference_seconds_sum':g.inference_seconds.sum(),
                                  'feature_seconds_sum':g.feature_seconds.sum() if 'feature_seconds' in g else 0,
                                  'max_sampled_worker_rss_MiB':g.sampled_peak_rss_bytes.max()/1024**2})
    pd.DataFrame(resource_rows).to_csv(run_dir/'selected_resource_comparison.csv',index=False)
    figure_dir=ROOT/'results/figures'/info['run_id'];figure_dir.mkdir(parents=True,exist_ok=True)
    names={'last_value':'Last value','weekly_seasonal_naive':'Weekly naive','ARIMA':'ARIMA','ETS':'ETS','XGBoost':'XGBoost'}
    palette=['#929aa4','#d5963a','#157c96','#8255a1','#158454']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    for (model,label),color in zip(names.items(),palette):
        g=table.loc[table.model.eq(model)].sort_values('horizon')
        for ax,metric in zip(axes,['RMSSE','MAE']):
            ax.plot(g.horizon,g[metric],marker='o',label=label,color=color)
            ax.set(xlabel='Forecast horizon (days)',ylabel=metric,xticks=[7,14,28]);ax.grid(alpha=.15)
            ax.spines[['top','right']].set_visible(False)
    axes[1].legend(frameon=False,fontsize=9)
    fig.suptitle('Stages 4–5 validation — selected models, identical scoring cases\nSales history only · lower is better · not final test performance',fontsize=12)
    fig.savefig(figure_dir/'validation_comparison.png',dpi=160);plt.close(fig)
    aggregate=table.groupby(['model','candidate'])[['MAE','RMSSE','WAPE']].mean().reset_index()
    formatted=aggregate.copy()
    for col in ['MAE','RMSSE','WAPE']:formatted[col]=formatted[col].map(lambda x:f'{x:.4f}')
    selection=pd.read_csv(run_dir/'candidate_selection.csv')[['candidate','selection_score','mae_tie_break','common_scored_cases','failed_cases']]
    for col in ['selection_score','mae_tie_break']:selection[col]=selection[col].map(lambda x:f'{x:.4f}')
    text=f'''# Stage 5 validation results — XGBoost

Run: `{info['run_id']}`. **Validation only. No final-test forecasts or scores.**

Completed {info['scheduled_pooled_jobs']} pooled jobs (3 candidates × 3 origins) on {info['series_count']} series. Each fit used {info['training_examples_per_series']*info['series_count']:,} examples, drawn entirely from its 365-day history window. Features are past sales only. There were {info['failed_pooled_jobs']} failed pooled jobs, affecting {info['affected_series_failures']} scheduled series cases. Full-run wall time: {info['elapsed_wall_seconds']:.1f} seconds; pooled fit timing is logged separately.

## Configuration selection

Selected: **`{selected['XGBoost']}`**. The fixed three-candidate search used equal 1/3 weights for 7/14/28-day macro RMSSE, with MAE tie-breaking. One fitted model shares information across all pilot series; identifiers are not predictors.

{markdown(selection)}

## Comparison with Stage 4

The previous Stage 4 choices are retained. All five methods are compared on the same {table.common_cases.min()}–{table.common_cases.max()} eligible series/origin cases per horizon, out of 600. Each number below equally averages the three horizon-level aggregates. WAPE is pooled within each horizon before that averaging. The table uses common RMSSE-eligible support for all three metrics; per-model full-coverage metrics are also saved.

{markdown(formatted)}

These are validation scores used to choose settings, not final-test estimates or evidence about pretrained models. The report does not declare a universal model-family winner. XGBoost's pooled learning differs from the per-series statistical fits; its raw-unit squared-error objective can emphasize larger-demand examples. No calendar or price information was supplied.

## Saved evidence

- `stage4_stage5_comparison.csv`: matched horizon-level scores.
- `comparison_by_segment.csv`: fixed demand/volume groups and store/category comparisons with support counts.
- `combined_reliability.csv`: median/p90/p95 and other per-series summaries; complete origins required per method.
- `combined_win_rates.csv`: common complete series across all five methods; tied wins are shared.
- `resources.csv` and `selected_resource_comparison.csv`: pooled versus per-series measurements explicitly labeled. Worker memory includes runtime overhead and is sampled; summed CPU-worker time is not wall time.
- `feature_gain.csv`: training-only tree gain importance, not causal effects.
- `selected_models.json`: selected configuration and hashes.
- `run.json`: immutable-at-run configuration, data/source hashes, software/hardware details, and history boundaries.

Forecasts: `results/forecasts/{info['run_id']}/`. Chart: `results/figures/{info['run_id']}/validation_comparison.png`.

Stage 4 source: `{old_info['run_id']}`. Its raw pilot, manifest, horizons, origins, and history condition were checked for compatibility before comparison. Neither its selections nor any locked protocol decision was changed.

## Next boundary

The primary Stage 5 implementation and validation are complete. Business-feature experiments and 90/180-day sensitivity remain separate later work. Stage 6 adds pretrained models. Final-test execution waits until all model-family and experiment choices are frozen.
'''
    (ROOT/'docs/stage5_results.md').write_text(text);(run_dir/'report.md').write_text(text)
    (run_dir/'comparison_provenance.json').write_text(json.dumps({'stage4_run':old_info['run_id'],
        'stage5_run':info['run_id'],'report_source_sha256':sha(Path(__file__)),
        'stage4_case_scores_sha256':sha(old_dir/'case_scores.csv'),
        'stage5_case_scores_sha256':sha(run_dir/'case_scores.csv')},indent=2)+'\n')
    print(formatted.to_string(index=False));print('Report:',ROOT/'docs/stage5_results.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run_dir',type=Path)
    write_report(parser.parse_args().run_dir)
