import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from src.models.xgboost_model import features_from_history, training_matrix, recursive_forecast, fit_model, feature_names
from src.evaluation.stage5 import run_pooled_task

ROOT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((ROOT/'configs/stage5.json').read_text())
FEATURES=CONFIG['features']


class FeatureTests(unittest.TestCase):
    def test_exact_lags_and_trailing_statistics(self):
        history=np.arange(1,29,dtype=float)[None,:]
        row=features_from_history(history,FEATURES)[0]
        np.testing.assert_array_equal(row[:5],[28,27,22,15,1])
        self.assertAlmostEqual(row[5],25)
        self.assertAlmostEqual(row[6],2)
        self.assertEqual(row[7],0)
        self.assertAlmostEqual(row[8],14.5)
        self.assertEqual(len(row),len(feature_names(FEATURES)))

    def test_training_target_is_not_in_its_features(self):
        history=np.arange(1,366,dtype=float)[None,:]
        x,y=training_matrix(history,FEATURES)
        changed=history.copy();changed[0,28]=999999
        x2,y2=training_matrix(changed,FEATURES)
        np.testing.assert_array_equal(x[0],x2[0])
        self.assertEqual(y[0],29);self.assertEqual(y2[0],999999)
        self.assertEqual(x.shape,(337,11));self.assertEqual(y[-1],365)
        self.assertEqual(x[-1,0],364)

    def test_windows_never_cross_series_boundaries(self):
        history=np.stack([np.ones(365),np.full(365,100)])
        x,y=training_matrix(history,FEATURES)
        np.testing.assert_array_equal(x[:337,0],np.ones(337))
        np.testing.assert_array_equal(x[337:,0],np.full(337,100))
        np.testing.assert_array_equal(y,np.r_[np.ones(337),np.full(337,100)])

    def test_feature_validation(self):
        for y in [np.ones((1,27)),np.array([[np.nan]*28]),-np.ones((1,28))]:
            with self.assertRaises(ValueError):features_from_history(y,FEATURES)

    def test_zero_history_is_valid(self):
        x,y=training_matrix(np.zeros((2,365)),FEATURES)
        self.assertTrue(np.isfinite(x).all());self.assertTrue((y==0).all())
        np.testing.assert_array_equal(x[:,7],np.ones(674))
        np.testing.assert_array_equal(x[:,10],np.ones(674))


class RecursiveTests(unittest.TestCase):
    def test_later_steps_use_own_predictions(self):
        class AddOne:
            def inplace_predict(self,x):return x[:,0]+1
        h=np.full((2,365),4.)
        p=recursive_forecast(AddOne(),h,28,FEATURES)
        np.testing.assert_array_equal(p[0],np.arange(5,33))
        np.testing.assert_array_equal(h,np.full((2,365),4.))

    def test_negative_raw_is_preserved_but_feedback_is_clipped(self):
        class MinusOne:
            def inplace_predict(self,x):return x[:,0]-1
        p=recursive_forecast(MinusOne(),np.zeros((1,365)),3,FEATURES)
        np.testing.assert_array_equal(p,[[-1,-1,-1]])

    def test_invalid_predictions_raise(self):
        class Bad:
            def inplace_predict(self,x):return np.full(len(x),np.nan)
        with self.assertRaises(ValueError):recursive_forecast(Bad(),np.ones((2,365)),28,FEATURES)

    def test_seeded_fit_reproduces_forecasts(self):
        history=np.stack([np.arange(50)%7,np.arange(50)%3]).astype(float)
        x,y=training_matrix(history,FEATURES)
        candidate={**CONFIG['candidates'][0],'num_boost_round':5}
        first=fit_model(x,y,candidate,CONFIG);second=fit_model(x,y,candidate,CONFIG)
        a=recursive_forecast(first,history,28,FEATURES);b=recursive_forecast(second,history,28,FEATURES)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(a.shape,(2,28))

    def test_pooled_fit_failure_stays_explicit(self):
        task={'settings':CONFIG,'history':np.ones((2,365)),'spec':CONFIG['candidates'][0],
              'forecast_origin':'2015-11-08','horizon':28}
        with patch('src.evaluation.stage5.fit_model',side_effect=RuntimeError('expected test failure')):
            result=run_pooled_task(task)
        self.assertEqual(result['status'],'failed');self.assertIsNone(result['raw'])
        self.assertIn('expected test failure',result['error'])

if __name__=='__main__':unittest.main()
