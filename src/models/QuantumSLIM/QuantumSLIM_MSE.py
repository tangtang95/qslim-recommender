import dimod
import numpy as np
from tqdm import tqdm
import scipy.sparse as sps

from course_lib.Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from course_lib.Base.Recommender_utils import check_matrix
from course_lib.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix
from src.models.QuantumSLIM.ResponseAggregator.ResponseAggregateStrategy import ResponseAggregateStrategy


def get_item_qubo_problem(URM_train, target_column):
    """
    Return a numpy matrix representing the QUBO problem of the MSE objective function of a single item ("item_index")

    :param URM_train: the csr matrix representing the training URM
    :param target_column: a numpy array containing the target item column
    :return: numpy matrix (item_size x item_size) containing the QUBO MSE problem
    """
    n_items = URM_train.shape[1]

    # Compute QUBO of sum over users of (sum over j of r_uj*s_ji)^2
    qubo = URM_train.T.dot(URM_train)

    # Compute QUBO of sum over users of (sum over j of r_ui*s_ji)^2
    qubo += np.ones(shape=(n_items, n_items), dtype=np.int) * (target_column.T.dot(target_column))

    # Compute QUBO of the double product between the previous summations
    qubo += - 2 * np.repeat(URM_train.T.dot(target_column)[:, np.newaxis], n_items, axis=1).reshape((n_items, n_items))
    return qubo


class QuantumSLIM_MSE(BaseItemSimilarityMatrixRecommender):
    """

    """
    RECOMMENDER_NAME = "QuantumSLIM_MSE"

    def __init__(self, URM_train, solver, agg_strategy: ResponseAggregateStrategy, verbose=True):
        super(QuantumSLIM_MSE, self).__init__(URM_train, verbose=verbose)
        self.solver = solver
        self.agg_strategy = agg_strategy

    def fit(self, topK=5, fixK=True, constraint_strength=-1, **solver_parameters):
        """

        :param topK:
        :param fixK:
        :param constraint_strength:
        :param solver_parameters:
        :return:
        """
        URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)
        n_items = URM_train.shape[1]
        matrix_builder = IncrementalSparseMatrix(n_rows=n_items, n_cols=n_items)
        mapping = {i: "s{}".format(i) for i in range(n_items)}

        for currentItem in tqdm(range(n_items), desc="Computing W_sparse matrix"):
            # get the target column
            target_column = URM_train[:, currentItem].toarray()

            # set the "currentItem"-th column of URM_train to zero
            start_pos = URM_train.indptr[currentItem]
            end_pos = URM_train.indptr[currentItem + 1]
            current_item_data_backup = URM_train.data[start_pos: end_pos].copy()
            URM_train.data[start_pos: end_pos] = 0.0

            # get BQM/QUBO problem for the current item
            qubo = get_item_qubo_problem(URM_train, target_column)
            if constraint_strength == -1:
                constraint_strength = np.max(qubo) - np.min(qubo)
            if topK != -1 and fixK:
                bqm = dimod.generators.combinations(n_items, topK, strength=constraint_strength)
                qubo = qubo + bqm.to_numpy_matrix()

            bqm = dimod.binary_quadratic_model.BQM.from_numpy_matrix(qubo, offset=0)
            bqm.relabel_variables(mapping=mapping)
            bqm.fix_variable("s{}".format(currentItem), 0)

            # solve the problem with the solver
            # response = self.solver.sample(bqm, chain_strength=constraint_strength, **solver_parameters)
            response = self.solver.sample(bqm, **solver_parameters)

            # aggregate response into a vector of similarities and save it into W_sparse
            solution_list = self.agg_strategy.get_aggregated_response(response)
            solution_list = np.insert(solution_list, 0, currentItem)
            matrix_builder.add_single_row(currentItem,  np.where(solution_list == 1)[0], data=1.0)

            # restore URM_train
            URM_train.data[start_pos:end_pos] = current_item_data_backup

        self.W_sparse = sps.csr_matrix(matrix_builder.get_SparseMatrix().T)
