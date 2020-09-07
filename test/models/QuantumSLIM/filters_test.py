import unittest

import pandas as pd
from src.models.QuantumSLIM.Filters.NoFilter import NoFilter
from src.models.QuantumSLIM.Filters.TopFilter import TopFilter


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        data = [[0, 1, 0, -20, 1], [0, 1, 1, -25, 1], [0, 0, 0, 0, 1]]
        self.df = pd.DataFrame(data=data, columns=["a00", "a01", "a02", "energy", "num_occurrences"])

    def test_no_filter_class(self):
        no_filter = NoFilter()
        new_df = no_filter.filter_samples(self.df)
        self.assertTrue(new_df.equals(self.df))

    def test_top_filter_class(self):
        top_filter = TopFilter(top_p=0.33)
        new_df = top_filter.filter_samples(self.df)

        self.assertTrue(new_df.shape[0] == 1)
        print(new_df["energy"].iloc[0])
        self.assertTrue(new_df["energy"].iloc[0] == -25)

        top_filter = TopFilter(top_p=0.66)
        new_df = top_filter.filter_samples(self.df)
        self.assertTrue(new_df.shape[0] == 2)


if __name__ == '__main__':
    unittest.main()
