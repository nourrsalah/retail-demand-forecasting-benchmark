"""Safety tests for historical sampling and join cardinality."""
import unittest
import pandas as pd
from src.data.pipeline import profile_series, select_pilot, build_long


class DataPipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = {'profile_days': 4, 'seed': 42, 'n_series': 3,
                       'intermittent_zero_fraction': .5, 'volatile_cv': 1.0}
        self.sales = pd.DataFrame([
            [f'item_{i}_store', f'item_{i}', 'dept', 'cat', 'store', 'state', *values]
            for i, values in enumerate([[2,2,2,2,0], [0,0,1,0,9], [1,1,1,20,3], [0,0,0,0,10]])
        ], columns=['id','item_id','dept_id','cat_id','store_id','state_id','d_1','d_2','d_3','d_4','d_5'])
        self.calendar = pd.DataFrame({'d': [f'd_{i}' for i in range(1,6)],
                                      'date': pd.date_range('2020-01-01', periods=5), 'wm_yr_wk': [1]*5})
        self.prices = pd.DataFrame({'item_id':['item_0'], 'store_id':['store'], 'wm_yr_wk':[1], 'sell_price':[2.]})

    def test_future_demand_cannot_change_selection(self):
        before = profile_series(self.sales, self.config)
        changed = self.sales.copy()
        changed['d_5'] = [10000,20000,0,999999]
        after = profile_series(changed, self.config)
        pd.testing.assert_frame_equal(before, after)
        pd.testing.assert_frame_equal(select_pilot(before, self.config), select_pilot(after, self.config))
        self.assertEqual(before.demand_type.tolist(), ['stable','intermittent','volatile','inactive_in_profile'])

    def test_merge_keeps_missing_prices_and_daily_targets(self):
        selected = select_pilot(profile_series(self.sales, self.config), self.config)
        long = build_long(self.sales, self.calendar, self.prices, selected)
        self.assertEqual(len(long), 15)
        self.assertEqual(int(long.price_missing.sum()), 10)
        self.assertEqual(long.units_sold.sum(), self.sales.iloc[:3, 6:].to_numpy().sum())

    def test_duplicate_price_keys_rejected(self):
        selected = select_pilot(profile_series(self.sales, self.config), self.config)
        with self.assertRaises(pd.errors.MergeError):
            build_long(self.sales, self.calendar, pd.concat([self.prices,self.prices]), selected)


if __name__ == '__main__':
    unittest.main()
