from scipy import sparse as sps
import numpy as np

from src.models.QuantumSLIM.Transformations.TransformationInterface import TransformationInterface


class NormMSETransformation(TransformationInterface):

    def get_qubo_problem(self, urm: sps.csr_matrix, target_column: np.ndarray):
        """
        Return a numpy matrix representing the QUBO problem of the normalized MSE objective function of a single item
        ("item_index")

        :param urm: the csr matrix representing the training URM
        :param target_column: a numpy array containing the target item column
        :return: numpy matrix (item_size x item_size) containing the QUBO MSE problem
        """
        n_items = urm.shape[1]

        # Compute QUBO of sum over users of (sum over j of r_uj*s_ji)^2
        qubo = urm.T.dot(urm)

        # Compute QUBO of sum over users of (sum over j of r_ui*s_ji)^2
        qubo += np.ones(shape=(n_items, n_items), dtype=np.int) * (target_column.T.dot(target_column))

        # Compute QUBO of the double product between the previous summations
        qubo += - 2 * np.repeat(urm.T.dot(target_column)[:, np.newaxis], n_items, axis=1).reshape(
            (n_items, n_items))
        return qubo
