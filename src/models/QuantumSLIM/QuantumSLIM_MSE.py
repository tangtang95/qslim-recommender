import dimod
import numpy as np
import scipy.sparse as sps
from dwave.system import EmbeddingComposite, LazyFixedEmbeddingComposite
from tqdm import tqdm

from course_lib.Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from course_lib.Base.Recommender_utils import check_matrix
from course_lib.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix
from src.models.QuantumSLIM.Filters.FilterStrategy import FilterStrategy
from src.models.QuantumSLIM.Filters.NoFilter import NoFilter
from src.models.QuantumSLIM.Aggregators.AggregatorInterface import AggregatorInterface
from src.models.QuantumSLIM.Aggregators.AggregatorFirst import AggregatorFirst
from src.models.QuantumSLIM.Losses.MSELoss import MSELoss
from src.models.QuantumSLIM.Losses.LossInterface import LossInterface


class QuantumSLIM_MSE(BaseItemSimilarityMatrixRecommender):
    """

    """

    RECOMMENDER_NAME = "QuantumSLIM_MSE"

    def __init__(self, URM_train, solver, agg_strategy: AggregatorInterface = AggregatorFirst(),
                 transform_fn: LossInterface = MSELoss(only_positive=False),
                 filter_strategy: FilterStrategy = NoFilter(), verbose=True):
        super(QuantumSLIM_MSE, self).__init__(URM_train, verbose=verbose)
        self.solver = solver
        self.agg_strategy = agg_strategy
        self.transform_fn = transform_fn
        self.filter_strategy = filter_strategy
        self.df_responses = None

    def build_similarity_matrix(self, df_responses):
        """

        :param df_responses:
        :return:
        """
        n_items = self.URM_train.shape[1]
        matrix_builder = IncrementalSparseMatrix(n_rows=n_items, n_cols=n_items)

        for currentItem in range(n_items):
            response_df = df_responses[df_responses.item_id == currentItem].copy()
            filtered_response_df = self.filter_strategy.filter_samples(response_df)
            solution_list = self.agg_strategy.get_aggregated_response(filtered_response_df)

            self._print("The aggregated response for item {} is {}".format(currentItem,
                                                                           solution_list))

            row_indices = np.where(solution_list > 0)[0]
            matrix_builder.add_data_lists(row_indices, [currentItem] * len(row_indices), solution_list[row_indices])
        return sps.csr_matrix(matrix_builder.get_SparseMatrix())

    def fit(self, topK=5, fixK=True, constraint_multiplier=1, chain_multiplier=1, remove_unpopular_items=False,
            unpopular_threshold=4, **solver_parameters):
        """

        :param topK:
        :param fixK:
        :param constraint_multiplier:
        :param chain_multiplier:
        :param remove_unpopular_items:
        :param unpopular_threshold:
        :param solver_parameters:
        :return:
        """
        URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)
        n_items = URM_train.shape[1]

        mapping = {i: "a{:02d}".format(i) for i in range(n_items)}
        min_constraint_strength = 10
        min_chain_strength = 10

        item_pop = np.array((URM_train > 0).sum(axis=0)).flatten()
        unpopular_items = np.where(item_pop <= unpopular_threshold)[0]

        for curr_item in tqdm(range(n_items), desc="{}: Computing W_sparse matrix".format(self.RECOMMENDER_NAME)):
            # get the target column
            target_column = URM_train[:, curr_item].toarray()

            # set the "curr_item"-th column of URM_train to zero
            start_pos = URM_train.indptr[curr_item]
            end_pos = URM_train.indptr[curr_item + 1]
            current_item_data_backup = URM_train.data[start_pos: end_pos].copy()
            URM_train.data[start_pos: end_pos] = 0.0

            # get BQM/QUBO problem for the current item
            qubo = self.transform_fn.get_qubo_problem(URM_train, target_column)
            if topK != -1 and fixK:
                constraint_strength = max(min_constraint_strength,
                                          constraint_multiplier * np.abs((np.max(qubo) - np.min(qubo))))
                bqm = dimod.generators.combinations(n_items, topK, strength=constraint_strength)
                qubo = qubo + bqm.to_numpy_matrix()

            bqm = dimod.binary_quadratic_model.BQM.from_numpy_matrix(qubo, offset=0)
            bqm.relabel_variables(mapping=mapping)

            if remove_unpopular_items:
                bqm.fix_variables({"a{:02d}".format(i): 0 for i in unpopular_items})

            self._print("The BQM for item {} is {}".format(curr_item, bqm))

            # solve the problem with the solver
            if type(self.solver) is EmbeddingComposite or type(self.solver) is LazyFixedEmbeddingComposite:
                chain_strength = max(min_chain_strength,
                                     chain_multiplier * np.abs((np.max(qubo) - np.min(qubo))))
                response = self.solver.sample(bqm, chain_strength=chain_strength, auto_scale=True, **solver_parameters)
            else:
                response = self.solver.sample(bqm, **solver_parameters)

            self._print("The response for item {} is {}".format(curr_item, response.aggregate()))
            if type(self.solver) is EmbeddingComposite or type(self.solver) is LazyFixedEmbeddingComposite:
                self._print("Break chain percentage of item {} is {}"
                            .format(curr_item, list(response.data(fields=["chain_break_fraction"]))))

            # save response in self.responses
            response_df = response.to_pandas_dataframe()
            response_df["a{:02d}".format(curr_item)] = 0.0
            if remove_unpopular_items:
                for i in unpopular_items:
                    response_df["a{:02d}".format(i)] = 0.0
            response_df["item_id"] = curr_item
            if self.df_responses is None:
                self.df_responses = response_df
                self.df_responses = self.df_responses.reindex(sorted(self.df_responses.columns), axis=1)
            else:
                self.df_responses = self.df_responses.append(response_df, ignore_index=True)

            # restore URM_train
            URM_train.data[start_pos:end_pos] = current_item_data_backup

        self.W_sparse = self.build_similarity_matrix(self.df_responses)
