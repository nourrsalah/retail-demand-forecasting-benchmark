"""Build reviewable Stage 4 validation tables and figures from a completed run."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.evaluation.metrics import pooled_wape
from src.evaluation.stage4 import ROOT


def write_report(run_dir):
    run_dir=Path(run_dir).resolve()
    info=json.loads((run_dir/'run.json').read_text())
    if info['state']!='complete' or info['smoke'] or info['split']!='validation':
        raise ValueError('Report requires a complete full validation run')
    selected=json.loads((run_dir/'selected_models.json').read_text())['selected_candidates']
    cases=pd.read_csv(run_dir/'case_scores.csv')
    resources=pd.read_csv(run_dir/'resources.csv')
    choices=pd.read_csv(run_dir/'candidate_selection.csv')
    retained=[v for v in selected.values() if v]
    g=cases.loc[cases.candidate.isin(retained)]
    key=['series_id','forecast_origin','horizon']
    # Matched support across all selected methods for the descriptive comparison.
    eligible=g.loc[g.status.eq('ok') & g.RMSSE.notna() & g.MAE.notna()]
    complete_keys=eligible.groupby(key).size()
    complete_keys=complete_keys[complete_keys.eq(len(retained))].index
    matched=eligible.set_index(key).loc[complete_keys].reset_index()
    rows=[]
    for (model,candidate,horizon), group in matched.groupby(['model','candidate','horizon']):
        rows.append({'model':model,'candidate':candidate,'horizon':horizon,'common_cases':len(group),
                     'MAE':group.MAE.mean(),'RMSSE':group.RMSSE.mean(),
                     'WAPE':pooled_wape(group.absolute_error_sum,group.actual_sum)})
    table=pd.DataFrame(rows)
    table.to_csv(run_dir/'selected_comparison_common_cases.csv',index=False)
    figure_dir=ROOT/'results/figures'/info['run_id']; figure_dir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    names={'last_value':'Last value','weekly_seasonal_naive':'Weekly naive','ARIMA':'ARIMA','ETS':'ETS'}
    palette={'last_value':'#8b96a3','weekly_seasonal_naive':'#d5963a','ARIMA':'#157c96','ETS':'#8255a1'}
    for model in names:
        v=table.loc[table.model.eq(model)].sort_values('horizon')
        for ax,metric in zip(axes,['RMSSE','MAE']):
            ax.plot(v.horizon,v[metric],marker='o',label=names[model],color=palette[model])
            ax.set(xlabel='Forecast horizon (days)',ylabel=metric,xticks=[7,14,28])
            ax.grid(alpha=.15)
    axes[1].legend(frameon=False)
    fig.suptitle('Stage 4 validation — selected configurations, common scoring cases\nLower is better · selection data, not final test performance',fontsize=12)
    fig.savefig(figure_dir/'validation_accuracy.png',dpi=160);plt.close(fig)
    reliability=pd.read_csv(run_dir/'reliability.csv')
    fig,ax=plt.subplots(figsize=(9,4.5),constrained_layout=True)
    order=[selected[m] for m in names if selected[m]]
    r=reliability.loc[reliability.horizon.eq(28)&reliability.metric.eq('RMSSE')].set_index('candidate').reindex(order)
    x=np.arange(len(order)); width=.23
    for offset,metric,color in zip([-1,0,1],['median','p90','p95'],['#157c96','#d5963a','#8255a1']):
        ax.bar(x+offset*width,r[metric],width,label=metric,color=color)
    ax.set(xticks=x,xticklabels=[names[m] for m in names if selected[m]],ylabel='Per-series RMSSE',
           title='28-day validation consistency across series\nScores averaged across all three origins; valid complete series only')
    ax.legend(frameon=False); ax.grid(axis='y',alpha=.15)
    fig.savefig(figure_dir/'validation_reliability.png',dpi=160);plt.close(fig)
    def markdown(frame):
        header='| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'
        return header+'\n'.join('| '+' | '.join(str(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None))
    means=table.groupby(['model','candidate'])[['MAE','RMSSE','WAPE']].mean().reset_index()
    for col in ['MAE','RMSSE','WAPE']:means[col]=means[col].map(lambda x:f'{x:.4f}')
    failures=resources.loc[resources.status.ne('ok')]
    overview=choices[['model','candidate','selection_score','common_scored_cases','failed_cases']].copy()
    overview.selection_score=overview.selection_score.map(lambda x:f'{x:.4f}')
    text=f'''# Stage 4 validation results

Run: `{info['run_id']}`. **Validation only; final test has not been evaluated.**

Completed {info['scheduled_tasks']:,} scheduled tasks across {info['series_count']} series, three validation origins, and eight prespecified candidates. Wall time: {info['elapsed_wall_seconds']:.1f} seconds with {info['workers']} CPU workers; this includes data preparation and artifact writing. Failed tasks: {info['failed_tasks']}. Predictions were not substituted for failures.

## Selected specifications

- ARIMA: `{selected['ARIMA']}`.
- ETS: `{selected['ETS']}`.
- Reference rules: last value and weekly seasonal naive.

Selection uses equal 1/3 horizon weights. Candidate scores below use common successful support **within each family**, so selection scores should not be compared across families without checking support.

{markdown(overview)}

## Matched descriptive comparison

This table averages each horizon's aggregate metric with equal weight. WAPE is first pooled within each horizon, never averaged across series. The underlying horizon table is `selected_comparison_common_cases.csv`. All selected methods share the same eligible series/origin cases at each horizon ({table.common_cases.min()}–{table.common_cases.max()} cases, out of 600). RMSSE eligibility also defines this table's MAE/WAPE support; full valid-MAE coverage is in `accuracy_by_horizon.csv`.

{markdown(means)}

These scores are used for tuning and do not constitute final-test evidence. They cannot establish conclusions about pretrained versus locally trained models, since pretrained models are not included yet.

## Reliability, failures, and resources

`reliability.csv` reports median/p90/p95/IQR/standard deviation/maximum across series with all three origins valid. `win_rates.csv` uses common complete series across selected methods and shares tied wins. These are validation-only descriptions. `origin_distributions.csv` exposes period-level variability.

Failures:

{markdown(failures[['candidate','series_id','forecast_origin','error']]) if len(failures) else 'None.'}

`resources.csv` records per-task fit, first inference, three-repeat warmed inference, warnings, and sampled process memory. `resource_summary.csv` summarizes worker measurements; summed worker seconds are not elapsed wall time. Hardware and measurement limitations are recorded in `run.json` and `docs/stage4.md`. Exact source/configuration hashes and software versions accompany the run.

## Files and next boundary

- Metrics and configuration selection: `results/metrics/{info['run_id']}/`.
- Candidate and selected forecasts: `results/forecasts/{info['run_id']}/`.
- Figures: `results/figures/{info['run_id']}/`.

The 365-day Stage 4 validation experiment is complete. Keep the statistical choices recorded while proceeding to other model families. Do not run the four final test origins until all relevant model and experiment choices are frozen. The 90/180-day sensitivity conditions remain later work.
'''
    (ROOT/'docs/stage4_results.md').write_text(text)
    (run_dir/'report.md').write_text(text)
    print(means.to_string(index=False))
    print('Report:',ROOT/'docs/stage4_results.md')
    print('Figures:',figure_dir)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    write_report(parser.parse_args().run_dir)
