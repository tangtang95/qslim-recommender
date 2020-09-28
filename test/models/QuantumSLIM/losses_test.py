import unittest
import scipy.sparse as sps
import numpy as np

from course_lib.Base.Recommender_utils import check_matrix
from src.models.QuantumSLIM.Losses.MSELoss import MSELoss
from src.models.QuantumSLIM.Losses.NormMSELoss import NormMSELoss


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.urm1 = sps.csr_matrix([[3, 2, 4], [1, 2, 1], [0, 1, 0]])

    def test_mse(self):
        curr_item = 0
        URM_train = check_matrix(self.urm1, 'csc', dtype=np.float32)
        target_column = URM_train[:, curr_item].toarray()

        start_pos = URM_train.indptr[curr_item]
        end_pos = URM_train.indptr[curr_item + 1]
        URM_train.data[start_pos: end_pos] = 0.0

        loss = MSELoss(only_positive=False)
        qubo = loss.get_qubo_problem(urm=URM_train, target_column=target_column)

        self.assertEqual(qubo.tolist(), [[0, 0, 0], [0, -7, 10], [0, 10, -9]])

    def test_non_zero_mse(self):
        curr_item = 0
        URM_train = check_matrix(self.urm1, 'csc', dtype=np.float32)
        target_column = URM_train[:, curr_item].toarray()

        start_pos = URM_train.indptr[curr_item]
        end_pos = URM_train.indptr[curr_item + 1]
        URM_train.data[start_pos: end_pos] = 0.0

        loss = MSELoss(only_positive=True)
        qubo = loss.get_qubo_problem(urm=URM_train, target_column=target_column)

        self.assertEqual(qubo.tolist(), [[0, 0, 0], [0, -8, 10], [0, 10, -9]])

    def test_norm_mse(self):
        curr_item = 0
        URM_train = check_matrix(self.urm1, 'csc', dtype=np.float32)
        target_column = URM_train[:, curr_item].toarray()

        start_pos = URM_train.indptr[curr_item]
        end_pos = URM_train.indptr[curr_item + 1]
        URM_train.data[start_pos: end_pos] = 0.0

        loss = NormMSELoss(only_positive=False, is_simplified=False)
        qubo = loss.get_qubo_problem(urm=URM_train, target_column=target_column)

        self.assertEqual(qubo.tolist(), [[10, 10, 10], [-6, 3, 4], [-16, -6, 1]])

    def test_non_zero_norm_mse(self):
        curr_item = 0
        URM_train = check_matrix(self.urm1, 'csc', dtype=np.float32)
        target_column = URM_train[:, curr_item].toarray()

        start_pos = URM_train.indptr[curr_item]
        end_pos = URM_train.indptr[curr_item + 1]
        URM_train.data[start_pos: end_pos] = 0.0

        loss = NormMSELoss(only_positive=True, is_simplified=False)
        qubo = loss.get_qubo_problem(urm=URM_train, target_column=target_column)

        self.assertEqual(qubo.tolist(), [[10, 10, 10], [-6, 2, 4], [-16, -6, 1]])

    def test_sim_norm_mse(self):
        curr_item = 0
        URM_train = check_matrix(self.urm1, 'csc', dtype=np.float32)
        target_column = URM_train[:, curr_item].toarray()

        start_pos = URM_train.indptr[curr_item]
        end_pos = URM_train.indptr[curr_item + 1]
        URM_train.data[start_pos: end_pos] = 0.0

        loss = NormMSELoss(only_positive=False, is_simplified=True)
        qubo = loss.get_qubo_problem(urm=URM_train, target_column=target_column)

        self.assertEqual(qubo.tolist(), [[0, 0, 0], [-16, -7, -6], [-26, -16, -9]])

    def test_non_zero_sim_norm_mse(self):
        curr_item = 0
        URM_train = check_matrix(self.urm1, 'csc', dtype=np.float32)
        target_column = URM_train[:, curr_item].toarray()

        start_pos = URM_train.indptr[curr_item]
        end_pos = URM_train.indptr[curr_item + 1]
        URM_train.data[start_pos: end_pos] = 0.0

        loss = NormMSELoss(only_positive=True, is_simplified=True)
        qubo = loss.get_qubo_problem(urm=URM_train, target_column=target_column)

        self.assertEqual(qubo.tolist(), [[0, 0, 0], [-16, -8, -6], [-26, -16, -9]])


if __name__ == '__main__':
    unittest.main()
