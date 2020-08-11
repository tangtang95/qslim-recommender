from scipy import sparse as sps
import numpy as np

from src.models.QuantumSLIM.Transformations.TransformationInterface import TransformationInterface


class MSETransformation(TransformationInterface):

    def get_qubo_problem(self, urm: sps.csr_matrix, target_column: np.ndarray):
        """
        Return a numpy matrix representing the QUBO problem of the not-normalized MSE objective function of a single
        item ("item_index")

        :param urm: the csr matrix representing the training URM
        :param target_column: a numpy array containing the target item column
        :return: numpy matrix (item_size x item_size) containing the QUBO MSE problem
        """
        return urm.T.dot(urm) - 2*np.diag(urm.T.dot(target_column))
