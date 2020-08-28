from abc import ABC, abstractmethod
import scipy.sparse as sps
import numpy as np


class TransformationInterface(ABC):

    @abstractmethod
    def get_qubo_problem(self, urm: sps.csr_matrix, target_column: np.ndarray):
        """
        Return a numpy matrix representing the QUBO problem of the objective function of a single item
        ("item_index")

        :param urm: the csr matrix representing the training URM
        :param target_column: a numpy array containing the target item column (user_size x 1)
        :return: numpy matrix (item_size x item_size) containing the QUBO MSE problem
        """
        pass
