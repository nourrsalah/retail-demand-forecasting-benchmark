import copy
import json
import tempfile
import unittest
from unittest.mock import Mock, patch
from contextlib import nullcontext
from pathlib import Path
import numpy as np
from src.models.foundation import FoundationAdapter, validate_history, validate_prediction
from src.evaluation.stage6 import ROOT, check_settings, verify_checkpoint
from src.evaluation.stage4 import sha


class Stage6Tests(unittest.TestCase):
    def test_history_boundary(self):
        x = np.arange(730).reshape(2,365)
        np.testing.assert_array_equal(validate_history(x),x)
        for bad in [x[:,:364],np.ones((2,366)),np.array([]),np.full((2,365),np.nan),-np.ones((2,365))]:
            with self.assertRaises(ValueError):validate_history(bad)

    def test_raw_negatives_retained_and_bad_forecasts_rejected(self):
        raw = -np.ones((2,28)); np.testing.assert_array_equal(validate_prediction(raw,2,28),raw)
        for bad in [np.zeros((28,2)),np.zeros((2,27)),np.full((2,28),np.inf)]:
            with self.assertRaises(ValueError):validate_prediction(bad,2,28)

    def test_locked_validation_guard(self):
        settings=json.loads((ROOT/'configs/stage6.json').read_text())
        protocol=json.loads((ROOT/'configs/evaluation.json').read_text())
        check_settings(settings,protocol)
        for key,value in [('split','test'),('scenario','business_features'),('history_days',366)]:
            changed={**settings,key:value}
            with self.assertRaises(ValueError):check_settings(changed,protocol)
        changed=copy.deepcopy(protocol);changed['validation_model_selection']['horizon_weights']['28']=.5
        with self.assertRaises(ValueError):check_settings(settings,changed)

    def test_timesfm_uses_mean_channel_not_returned_median(self):
        adapter=FoundationAdapter.__new__(FoundationAdapter)
        adapter.spec={'id':'timesfm25'};adapter.settings={'seed':42}
        full=np.full((2,28,10),3.0);full[...,0]=9.0
        adapter.model=Mock();adapter.model.forecast.return_value=(full[...,5],full)
        torch=Mock();torch.inference_mode.return_value=nullcontext()
        with patch.dict('sys.modules',{'torch':torch}):
            predicted=adapter.predict(np.ones((2,365)),'2015-11-08')
        np.testing.assert_array_equal(predicted,np.full((2,28),9.0))

    def test_checkpoint_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'model.safetensors').write_bytes(b'expected')
            spec={'id':'example','repo_id':'org/model'}
            lock={'models':{'example':{'repo_id':'org/model','revision':'a'*40,'snapshot_path':'.',
                'files':{'model.safetensors':{'sha256':sha(root/'model.safetensors')}}}}}
            self.assertEqual(verify_checkpoint(root,spec,lock),root)
            (root/'model.safetensors').write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_checkpoint(root,spec,lock)


if __name__=='__main__':unittest.main()
