import unittest

import pandas as pd
import numpy as np

from src.models.QuantumSLIM.Aggregators.AggregatorFirst import AggregatorFirst

from src.models.QuantumSLIM.Aggregators.AggregatorUnion import AggregatorUnion


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        data1 = [[0, 1, 0, -20, 1], [0, 1, 1, -25, 1], [0, 0, 0, 0, 1]]
        self.df1 = pd.DataFrame(data=data1, columns=["a00", "a01", "a02", "energy", "num_occurrences"])
        self.log_operation_fn = lambda arr: np.log1p(arr)
        self.no_operation_fn = lambda arr: arr
        self.exp_operation_fn = lambda arr: np.exp(arr)

        data2 = [[0, 1, 0, -2, 1], [0, 1, 1, -1, 1], [0, 0, 0, 0, 1]]
        self.df2 = pd.DataFrame(data=data2, columns=["a00", "a01", "a02", "energy", "num_occurrences"])

    def test_aggregator_first_class(self):
        agg_first = AggregatorFirst()
        res = agg_first.get_aggregated_response(self.df1)

        self.assertTrue(res.tolist() == [0, 1, 1])

    def test_aggregator_union_class(self):
        agg_avg = AggregatorUnion(self.no_operation_fn, is_filter_first=False, is_weighted=False)
        res = agg_avg.get_aggregated_response(self.df1)
        self.assertTrue(res.tolist() == [0, 2/3, 1/3])

        agg_avg_first = AggregatorUnion(self.no_operation_fn, is_filter_first=True, is_weighted=False)
        res = agg_avg_first.get_aggregated_response(self.df2)
        self.assertTrue(res.tolist() == [0, 2/3, 0])

        agg_weighted_avg = AggregatorUnion(self.no_operation_fn, is_filter_first=False, is_weighted=True)
        res = agg_weighted_avg.get_aggregated_response(self.df2)
        self.assertTrue(res.tolist() == [0, 1.5/1.5, 0.5/1.5])


if __name__ == '__main__':
    unittest.main()
