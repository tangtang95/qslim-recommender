from scipy import sparse as sps
import numpy as np

from src.models.QuantumSLIM.Losses.LossInterface import LossInterface


class MSELoss(LossInterface):
    def __init__(self, only_positive: bool):
        self.only_positive = only_positive

    def get_qubo_problem(self, urm: sps.csr_matrix, target_column: np.ndarray):
        """
        Return a numpy matrix representing the QUBO problem of the not-normalized MSE objective function of a single
        item ("item_index")

        :param urm: the csr matrix representing the training URM
        :param target_column: a numpy array containing the target item column (user_size x 1)
        :return: numpy matrix (item_size x item_size) containing the QUBO MSE problem
        """
        t_urm = urm
        t_target = target_column
        if self.only_positive:
            t_urm = urm[(target_column > 0).ravel(), :]
            t_target = target_column[target_column > 0]

        return t_urm.T.dot(t_urm) - 2*np.diag(t_urm.T.dot(t_target))
