import dimod
import numpy as np
import scipy.sparse as sps
from dwave.system import EmbeddingComposite
from tqdm import tqdm

from course_lib.Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from course_lib.Base.Recommender_utils import check_matrix
from course_lib.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix
from src.models.QuantumSLIM.ResponseAggregators.ResponseAggregateStrategy import ResponseAggregateStrategy
from src.models.QuantumSLIM.ResponseAggregators.ResponseFirst import ResponseFirst
from src.models.QuantumSLIM.Transformations.MSETransformation import MSETransformation
from src.models.QuantumSLIM.Transformations.TransformationInterface import TransformationInterface


class QuantumSLIM_MSE(BaseItemSimilarityMatrixRecommender):
    """

    """
    RECOMMENDER_NAME = "QuantumSLIM_MSE"

    def __init__(self, URM_train, solver, agg_strategy: ResponseAggregateStrategy = ResponseFirst(),
                 transform_fn: TransformationInterface = MSETransformation(), verbose=True):
        super(QuantumSLIM_MSE, self).__init__(URM_train, verbose=verbose)
        self.solver = solver
        self.agg_strategy = agg_strategy
        self.transform_fn = transform_fn

    def fit(self, topK=5, fixK=True, constraint_multiplier=1, chain_multiplier=1, **solver_parameters):
        """

        :param topK:
        :param fixK:
        :param constraint_multiplier:
        :param chain_multiplier:
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
            qubo = self.transform_fn.get_qubo_problem(URM_train, target_column)
            constraint_strength = constraint_multiplier*(np.max(qubo) - np.min(qubo))
            if topK != -1 and fixK:
                bqm = dimod.generators.combinations(n_items, topK, strength=constraint_strength)
                qubo = qubo + bqm.to_numpy_matrix()

            bqm = dimod.binary_quadratic_model.BQM.from_numpy_matrix(qubo, offset=0)
            bqm.relabel_variables(mapping=mapping)
            bqm.fix_variable("s{}".format(currentItem), 0)

            self._print("The BQM for item {} is {}".format(currentItem, bqm))

            # solve the problem with the solver
            if type(self.solver) is EmbeddingComposite:
                chain_strength = chain_multiplier*(np.max(qubo) - np.min(qubo))
                response = self.solver.sample(bqm, chain_strength=chain_strength, **solver_parameters)
            else:
                response = self.solver.sample(bqm, **solver_parameters)

            self._print("The response for item {} is {}".format(currentItem, response.aggregate()))
            if type(self.solver) is EmbeddingComposite:
                self._print("Break chain percentage of item {} is {}"
                            .format(currentItem, list(response.data(fields=["chain_break_fraction"]))))

            # aggregate response into a vector of similarities and save it into W_sparse
            solution_list = self.agg_strategy.get_aggregated_response(response)
            solution_list = np.insert(solution_list, currentItem, 0)

            self._print("The aggregated response for item {} is {}".format(currentItem,
                                                                               solution_list))
            row_indices = np.where(solution_list > 0)[0]
            matrix_builder.add_data_lists(row_indices, [currentItem]*len(row_indices), solution_list[row_indices])

            # restore URM_train
            URM_train.data[start_pos:end_pos] = current_item_data_backup

        self.W_sparse = sps.csr_matrix(matrix_builder.get_SparseMatrix())
