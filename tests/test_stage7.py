import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.evaluation.stage7 import audit_forecasts, bootstrap_comparison
from src.evaluation.benchmark import verify_freeze
from src.models.business_xgboost import business_features
from src.models.foundation import validate_history


class SharedEvaluationTests(unittest.TestCase):
    def frame(self):
        rows=[]
        for h in [7,14,28]:
            for lead in range(1,h+1):rows.append(dict(candidate='a',series_id='s',forecast_origin='2015-11-08',horizon=h,lead=lead,
                scenario='univariate',history_window=365,status='ok',date=str((pd.Timestamp('2015-11-08')+pd.Timedelta(days=lead)).date()),y_true=2.,y_pred_raw=1.,y_pred=1.))
        return pd.DataFrame(rows)

    def test_schedule_and_prefix_guards(self):
        frame=self.frame();audit_forecasts(frame,['s'],['2015-11-08'],['a'])
        for bad in [frame.iloc[1:].copy(),pd.concat([frame,frame.iloc[:1]])]:
            with self.assertRaises(ValueError):audit_forecasts(bad,['s'],['2015-11-08'],['a'])
        bad=frame.copy();bad.loc[0,'y_pred']=3
        with self.assertRaises(ValueError):audit_forecasts(bad,['s'],['2015-11-08'],['a'])
        bad=frame.copy();bad.loc[0,'date']='2016-02-01'
        with self.assertRaises(ValueError):audit_forecasts(bad,['s'],['2015-11-08'],['a'])

    def test_paired_bootstrap_constant_difference(self):
        rows=[]
        for model,delta in [('a',0),('b',2)]:
            for sid in ['s1','s2','s3']:
                for origin in ['o1','o2','o3']:
                    for h in [7,14,28]:rows.append(dict(model=model,series_id=sid,forecast_origin=origin,horizon=h,status='ok',MAE=1+delta,RMSSE=1+delta,absolute_error_sum=h*(1+delta),actual_sum=h))
        result=bootstrap_comparison(pd.DataFrame(rows),{'seed':42,'replicates':100,'confidence_level':.95})
        np.testing.assert_allclose(result[['difference_a_minus_b','ci_lower','ci_upper']],-2)

    def test_freeze_detects_changed_sources(self):
        freeze={'status':'frozen_before_final_test','source_sha256':{'x':'old'},'validation_artifacts_sha256':{}}
        with patch('src.evaluation.benchmark.source_hashes',return_value={'x':'new'}):
            with self.assertRaises(ValueError):verify_freeze(freeze)

    def test_short_foundation_context_is_exact(self):
        for days in [90,180,365]:self.assertEqual(validate_history(np.ones((2,days)),days).shape,(2,days))
        with self.assertRaises(ValueError):validate_history(np.ones((2,91)),90)

    def test_business_future_sales_and_prices_cannot_change_features(self):
        history=pd.DataFrame({'date':pd.date_range('2015-01-01',periods=4),'sell_price':[np.nan,2,3,4],
            'event_name_1':[None]*4,'event_name_2':[None]*4,'snap_CA':[0]*4})
        future=pd.DataFrame({'date':pd.date_range('2015-01-05',periods=2),'sell_price':[999,999],
            'units_sold':[999,999],'event_name_1':[None]*2,'event_name_2':[None]*2,'snap_CA':[0]*2})
        past,a=business_features(history,future,'CA');future[['sell_price','units_sold']]=-999
        _,b=business_features(history,future,'CA');np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(a[:,-2],4);np.testing.assert_array_equal(past[:,-2],[0,0,2,3])


if __name__=='__main__':unittest.main()
