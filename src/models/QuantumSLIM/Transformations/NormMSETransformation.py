from scipy import sparse as sps
import numpy as np

from src.models.QuantumSLIM.Transformations.TransformationInterface import TransformationInterface


class NormMSETransformation(TransformationInterface):

    def __init__(self, only_positive: bool, is_simplified: bool):
        self.only_positive = only_positive
        self.is_simplified = is_simplified

    def get_qubo_problem(self, urm: sps.csr_matrix, target_column: np.ndarray):
        """
        Return a numpy matrix representing the QUBO problem of the normalized MSE objective function of a single item
        ("item_index")

        :param urm: the csr matrix representing the training URM
        :param target_column: a numpy array containing the target item column (user_size x 1)
        :return: numpy matrix (item_size x item_size) containing the QUBO MSE problem
        """
        t_urm = urm
        t_target = target_column
        if self.only_positive:
            t_urm = urm[(target_column > 0).ravel(), :]
            t_target = target_column[target_column > 0]

        n_items = t_urm.shape[1]

        # Compute QUBO of sum over users of (sum over j of r_uj*s_ji)^2
        qubo = t_urm.T.dot(t_urm)

        # Compute QUBO of sum over users of (sum over j of r_ui*s_ji)^2
        if not self.is_simplified:
            qubo += np.ones(shape=(n_items, n_items), dtype=np.int) * (t_target.T.dot(t_target))

        # Compute QUBO of the double product between the previous summations
        qubo += - 2 * np.repeat(t_urm.T.dot(t_target)[:, np.newaxis], n_items, axis=1).reshape(
            (n_items, n_items))
        return qubo
