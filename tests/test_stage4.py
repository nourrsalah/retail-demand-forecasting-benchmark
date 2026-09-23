import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.models.statistical import fit_model, predict, NonConvergenceError
from src.evaluation.metrics import score, rmsse_scale, pooled_wape, equal_horizon_score
from src.evaluation.windows import make_window, validation_data
from src.evaluation.stage4 import select_candidates


class MetricTests(unittest.TestCase):
    def test_known_errors_and_training_scale(self):
        result=score([0,0,2,4,6],[0,4],[1,2])
        self.assertAlmostEqual(result['MAE'][0],1.5)
        self.assertAlmostEqual(result['WAPE'][0],.75)
        self.assertAlmostEqual(result['RMSSE'][0],np.sqrt(2.5/4))

    def test_undefined_scales_and_zero_demand(self):
        for y in [[0,0,0],[0,0,1],[0,2,2]]:
            self.assertTrue(np.isnan(rmsse_scale(y)[0]))
        self.assertTrue(np.isnan(score([1,2,1],[0,0],[1,1])['WAPE'][0]))
        self.assertEqual(score([1,2,1],[0,0],[1,1])['MAE'][0],1)

    def test_equal_horizon_weights_and_missing_horizon(self):
        self.assertEqual(equal_horizon_score({7:1,14:2,28:9}),4)
        self.assertTrue(np.isnan(equal_horizon_score({7:1,14:np.nan,28:9})))
        with self.assertRaises(ValueError): equal_horizon_score({7:1,28:9})

    def test_wape_pools_volume(self):
        self.assertAlmostEqual(pooled_wape([1,10],[1,100]),11/101)
        self.assertTrue(np.isnan(pooled_wape([3],[0])) )


class ForecastTests(unittest.TestCase):
    def test_baselines_do_not_update_with_actuals(self):
        y=np.arange(1,15,dtype=float)
        np.testing.assert_array_equal(predict(fit_model(y,{'model':'last_value'}),28),np.full(28,14))
        np.testing.assert_array_equal(predict(fit_model(y,{'model':'weekly_seasonal_naive'}),28),np.tile(y[-7:],4))

    def test_windows_exclude_future_actuals(self):
        s=pd.Series(np.arange(50),index=pd.date_range('2020-01-01',periods=50))
        a,b,d=make_window(s,'2020-01-20',10,7)
        changed=s.copy(); changed.loc['2020-01-21':]=99999
        c,_,_=make_window(changed,'2020-01-20',10,7)
        np.testing.assert_array_equal(a,c)
        np.testing.assert_array_equal(a,np.arange(10,20))
        self.assertEqual(d[0],pd.Timestamp('2020-01-21'))
        with self.assertRaises(ValueError):make_window(s.drop(pd.Timestamp('2020-01-19')),'2020-01-20',10,7)

    def test_final_test_rows_removed(self):
        p=pd.DataFrame({'series_id':['a']*3,'date':pd.to_datetime(['2016-01-03','2016-01-31','2016-02-01']),'units_sold':[1,2,999]})
        result=validation_data(p,{'validation_origins':['2016-01-03'],'horizons_days':[7,14,28]})
        self.assertEqual(len(result),2)

    def test_nonconvergence_is_not_a_fallback(self):
        spec={'model':'ARIMA','order':[1,0,0],'seasonal_order':[0,0,0,0],'trend':'c'}
        with patch('src.models.statistical.ARIMA') as model:
            model.return_value.fit.return_value.mle_retvals={'converged':False}
            with self.assertRaises(NonConvergenceError):fit_model(np.arange(1,20),spec)

    def test_statistical_adapters_return_finite_28_day_forecast(self):
        rng=np.random.default_rng(42)
        y=10+np.sin(np.arange(365)*2*np.pi/7)+rng.normal(0,.3,365)
        specs=[{'model':'ARIMA','order':[1,0,0],'seasonal_order':[0,0,0,0],'trend':'c'},
               {'model':'ETS','error':'add','trend':None,'seasonal':'add','seasonal_periods':7,'damped_trend':False}]
        for spec in specs:
            raw=predict(fit_model(y,spec),28)
            self.assertEqual(raw.shape,(28,)); self.assertTrue(np.isfinite(raw).all())


class SelectionTests(unittest.TestCase):
    def test_failed_candidate_does_not_get_easier_comparison(self):
        rows=[]
        for candidate in ['a','b']:
            for sid in ['easy','hard']:
                for h in [7,14,28]:
                    failed=candidate=='b' and sid=='hard'
                    rows.append({'candidate':candidate,'series_id':sid,'forecast_origin':'2020-01-01',
                                 'horizon':h,'status':'failed' if failed else 'ok','scale_valid':True,
                                 'RMSSE':np.nan if failed else (1 if candidate=='a' else 2),
                                 'MAE':np.nan if failed else 1})
        ranks,selected=select_candidates(pd.DataFrame(rows),[{'id':c,'model':'ARIMA'} for c in ['a','b']],[7,14,28])
        self.assertEqual(selected['ARIMA'],'a')
        self.assertTrue(ranks.common_successful_cases.eq(1).all())
        self.assertEqual(ranks.set_index('candidate').loc['b','failure_rate'],.5)

if __name__=='__main__': unittest.main()
