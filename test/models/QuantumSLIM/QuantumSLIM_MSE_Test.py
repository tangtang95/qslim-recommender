import unittest
import scipy.sparse as sps
import neal
import numpy as np

from src.models.QuantumSLIM.QuantumSLIM_MSE import QuantumSLIM_MSE
from src.models.QuantumSLIM.ResponseAggregator.ResponseFirstStrategy import ResponseFirstStrategy


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.URM = sps.random(20, 10, density=0.5, format="csr", dtype=int, random_state=123, data_rvs=np.ones)
        solver = neal.SimulatedAnnealingSampler()
        self.model = QuantumSLIM_MSE(URM_train=self.URM, solver=solver, agg_strategy=ResponseFirstStrategy())

    def test_fit_model(self):
        print()
        print(self.URM.todense())
        self.model.fit(topK=2, num_reads=10)
        print(self.model.W_sparse.todense())


if __name__ == '__main__':
    unittest.main()
