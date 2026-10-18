"""Use exactly the same successful targets for all history-length comparisons."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.evaluation.stage4 import ROOT, sha
from src.evaluation.metrics import pooled_wape
from src.evaluation.report_stage5 import markdown


def run():
    freeze=json.loads((ROOT/'configs/stage7_freeze.json').read_text())
    if freeze['auxiliary_analysis_sha256']['analysis/stage7_history.py']!=sha(Path(__file__)):
        raise ValueError('History analysis changed after its pre-test freeze')
    out=ROOT/'results/metrics'/('stage7-final-'+sha(ROOT/'configs/stage7_freeze.json')[:12])
    cases=pd.read_csv(out/'case_scores.csv');cases=cases.loc[cases.scenario.eq('univariate')].copy()
    keys=['series_id','forecast_origin','horizon'];expected=cases.model.nunique()*cases.history_window.nunique()
    if cases.duplicated(['model','history_window']+keys).any():raise ValueError('Duplicate history comparison case')
    good=cases.loc[cases.status.eq('ok') & cases.MAE.notna()]
    counts=good.groupby(keys).size();common=counts[counts.eq(expected)].index
    good=good.set_index(keys).loc[common].reset_index()
    rows=[]
    for (model,history,horizon),g in good.groupby(['model','history_window','horizon']):
        rows.append({'model':model,'history_days':history,'horizon':horizon,'common_cases':len(g),
                     'MAE':g.MAE.mean(),'WAPE':pooled_wape(g.absolute_error_sum,g.actual_sum)})
    table=pd.DataFrame(rows);table.to_csv(out/'history_identical_targets_by_horizon.csv',index=False)
    summary=table.groupby(['model','history_days'])[['MAE','WAPE']].mean().reset_index()
    summary.to_csv(out/'history_identical_targets_summary.csv',index=False)
    formatted=summary.copy()
    for col in ['MAE','WAPE']:formatted[col]=formatted[col].map(lambda x:f'{x:.4f}')
    for path in [out/'report.md',ROOT/'docs/stage7_final_results.md']:
        text=path.read_text();start=text.index('## History sensitivity');end=text.index('## Separate business-feature experiment')
        section='## History sensitivity\n\n'+markdown(formatted)+'\n\n'
        section+=f'This table uses identical successful series/origin targets across all eight methods and all three history lengths: {table.common_cases.min()}–{table.common_cases.max()} cases per horizon out of 800. MAE and pooled WAPE are averaged with equal horizon weights. Zero-scale histories remain eligible for these unscaled metrics. Failures excluded from this matched comparison remain in the coverage table.\n\n'
        section+='RMSSE has a history-specific denominator and is used only for model comparisons within a history. The separate per-history summaries can use different RMSSE-eligible support; they are not the basis for this shared-target sensitivity table. See `history_identical_targets_*.csv`.\n\n'
        path.write_text(text[:start]+section+text[end:])
    provenance=json.loads((out/'provenance.json').read_text())
    provenance['history_analysis_sha256']=sha(Path(__file__));provenance['history_common_cases_per_horizon']=table.groupby('horizon').common_cases.first().to_dict()
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(summary.to_string(index=False));print('Identical-target history comparison verified.')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--freeze',action='store_true')
    if parser.parse_args().freeze:
        path=ROOT/'configs/stage7_freeze.json'
        if (ROOT/'results/metrics/stage7_test_index.json').exists():raise ValueError('Cannot change the analysis freeze after test execution')
        frozen=json.loads(path.read_text())
        frozen['auxiliary_analysis_sha256']={'analysis/stage7_history.py':sha(Path(__file__))}
        path.write_text(json.dumps(frozen,indent=2)+'\n')
        print('Identical-target history analysis frozen before test.')
    else:run()
