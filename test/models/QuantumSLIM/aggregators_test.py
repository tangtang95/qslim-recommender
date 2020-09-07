import unittest

import pandas as pd
import numpy as np

from src.models.QuantumSLIM.ResponseAggregators.ResponseFirst import ResponseFirst


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        data = [[0, 1, 0, -20, 1], [0, 1, 1, -25, 1], [0, 0, 0, 0, 1]]
        self.df = pd.DataFrame(data=data, columns=["a00", "a01", "a02", "energy", "num_occurrences"])

    def test_response_first_class(self):
        agg_first = ResponseFirst()
        res = agg_first.get_aggregated_response(self.df)

        self.assertTrue(res.tolist() == [0, 1, 1])

    def test_response_generic_operation_class(self):
        pass  # TODO


if __name__ == '__main__':
    unittest.main()
