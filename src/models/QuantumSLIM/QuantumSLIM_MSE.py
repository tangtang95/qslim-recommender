import traceback

import dimod
import numpy as np
import pandas as pd
import scipy.sparse as sps
from tqdm import tqdm

from course_lib.Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from course_lib.Base.Recommender_utils import check_matrix
from course_lib.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix
from src.models.QuantumSLIM.Aggregators.AggregatorFirst import AggregatorFirst
from src.models.QuantumSLIM.Aggregators.AggregatorUnion import AggregatorUnion
from src.models.QuantumSLIM.Filters.NoFilter import NoFilter
from src.models.QuantumSLIM.Filters.TopFilter import TopFilter
from src.models.QuantumSLIM.ItemSelectors.ItemSelectorAll import ItemSelectorAll
from src.models.QuantumSLIM.ItemSelectors.ItemSelectorByCosineSimilarity import ItemSelectorByCosineSimilarity
from src.models.QuantumSLIM.ItemSelectors.ItemSelectorByEntropy import ItemSelectorByEntropy
from src.models.QuantumSLIM.ItemSelectors.ItemSelectorByPopularity import ItemSelectorByPopularity
from src.models.QuantumSLIM.ItemSelectors.ItemSelectorByVariance import ItemSelectorByVariance
from src.models.QuantumSLIM.Losses.MSELoss import MSELoss
from src.models.QuantumSLIM.Losses.NormMSELoss import NormMSELoss
from src.models.QuantumSLIM.Losses.NormMeanErrorLoss import NormMeanErrorLoss


class QuantumSLIM_MSE(BaseItemSimilarityMatrixRecommender):
    """
    It trains a Sparse LInear Methods (SLIM) item similarity model by using a Quantum DWave Machine. The objective
    function are MSE losses and some variants with some other regulators.
    """

    RECOMMENDER_NAME = "QuantumSLIM_MSE"

    MIN_CONSTRAINT_STRENGTH = 10
    ITEM_ID_COLUMN_NAME = "item_id"

    AGGREGATORS = {
        "FIRST": AggregatorFirst(),
        "LOG": AggregatorUnion(lambda arr: np.log1p(arr), is_filter_first=False, is_weighted=False),
        "LOG_FIRST": AggregatorUnion(lambda arr: np.log1p(arr), is_filter_first=True, is_weighted=False),
        "EXP": AggregatorUnion(lambda arr: np.exp(arr), is_filter_first=False, is_weighted=False),
        "EXP_FIRST": AggregatorUnion(lambda arr: np.exp(arr), is_filter_first=True, is_weighted=False),
        "AVG": AggregatorUnion(lambda arr: arr, is_filter_first=False, is_weighted=False),
        "AVG_FIRST": AggregatorUnion(lambda arr: arr, is_filter_first=True, is_weighted=False),
        "WEIGHTED_AVG": AggregatorUnion(lambda arr: arr, is_filter_first=False, is_weighted=True),
        "WEIGHTED_AVG_FIRST": AggregatorUnion(lambda arr: arr, is_filter_first=True, is_weighted=True)
    }

    LOSSES = {
        "MSE": MSELoss(only_positive=False),
        "NORM_MSE": NormMSELoss(only_positive=False, is_simplified=False),
        "NON_ZERO_MSE": MSELoss(only_positive=True),
        "NON_ZERO_NORM_MSE": NormMSELoss(only_positive=True, is_simplified=False),
        "SIM_NORM_MSE": NormMSELoss(only_positive=False, is_simplified=True),
        "SIM_NON_ZERO_NORM_MSE": NormMSELoss(only_positive=True, is_simplified=True),
        "NORM_MEAN_ERROR": NormMeanErrorLoss(only_positive=False, is_squared=False),
        "NORM_MEAN_ERROR_SQUARED": NormMeanErrorLoss(only_positive=False, is_squared=True)
    }

    FILTER_SAMPLES_METHODS = {
        "NONE": NoFilter(),
        "TOP_50_PERCENT": TopFilter(0.5),
        "TOP_20_PERCENT": TopFilter(0.2)
    }

    FILTER_ITEMS_METHODS = {
        "NONE": ItemSelectorAll(),
        "POPULARITY": ItemSelectorByPopularity(),
        "ABSOLUTE_ENTROPY": ItemSelectorByEntropy(),
        "VARIANCE": ItemSelectorByVariance(),
        "COSINE": ItemSelectorByCosineSimilarity(topK=10000, shrink=100, normalize=True)
    }

    def __init__(self, URM_train, solver: dimod.Sampler, obj_function: str = "NORM_MSE",
                 do_save_responses=False, verbose=True):
        super(QuantumSLIM_MSE, self).__init__(URM_train, verbose=verbose)
        self._check_init_parameters(obj_function)

        self.obj_function = obj_function
        self.solver = solver
        self.df_responses = None
        self.mapping_matrix = []
        self.do_save_responses = do_save_responses
        self.to_resume = False

    @classmethod
    def get_implemented_aggregators(cls):
        return list(cls.AGGREGATORS.keys())

    @classmethod
    def get_implemented_losses(cls):
        return list(cls.LOSSES.keys())

    @classmethod
    def get_implemented_filter_samples_methods(cls):
        return list(cls.FILTER_SAMPLES_METHODS.keys())

    @classmethod
    def get_implemented_filter_item_methods(cls):
        return list(cls.FILTER_ITEMS_METHODS.keys())

    def _check_init_parameters(self, obj_function):
        if obj_function not in self.get_implemented_losses():
            raise NotImplementedError("Objective function {} is not implemented".format(obj_function))

    def _check_fit_parameters(self, aggregation_strategy, filter_item_method, filter_sample_method):
        if aggregation_strategy not in self.get_implemented_aggregators():
            raise NotImplementedError("Filter strategy {} is not implemented".format(aggregation_strategy))
        if filter_item_method not in self.get_implemented_filter_item_methods():
            raise NotImplementedError("Filter item method {} is not implemented".format(filter_item_method))
        if filter_sample_method not in self.get_implemented_filter_samples_methods():
            raise NotImplementedError("Filter sample method {} is not implemented".format(filter_sample_method))

    def preload_fit(self, df_responses, mapping_matrix):
        """
        Preload the dataframe of df_responses to prepare for the fit function in case you need to resume the fit
        function failed previously.

        :param df_responses: dataframe with all the samples collected by the crashed fit function
        :param mapping_matrix: list containing the mapping of each item problem into its original variables
        """
        self.df_responses = df_responses
        self.mapping_matrix = mapping_matrix
        self.to_resume = True

    def add_sample_responses_to_matrix_builder(self, matrix_builder, agg_strategy, filter_sample_method,
                                               response_df, curr_item, mapping):
        """
        Add a column on "curr_item" index to the IncrementalSparseMatrix matrix_builder with the samples inside
        response_df.

        :param matrix_builder: The IncrementalSparseMatrix builder required to build the sparse matrix
        :param agg_strategy: the post-processing aggregation to be used on the samples
        :param filter_sample_method: the filter technique used before the post-processing aggregation
        :param response_df: the samples related to the item of index "curr_item"
        :param curr_item: the index of the item column to be added on the builder
        :type mapping: np.ndarray containing the mapping of the samples variables into the original variables
        :return: None
        """
        filtered_response_df = self.FILTER_SAMPLES_METHODS[filter_sample_method].filter_samples(response_df)
        solution_list = self.AGGREGATORS[agg_strategy].get_aggregated_response(filtered_response_df)

        self._print("The aggregated response for item {} is {}".format(curr_item,
                                                                       solution_list))

        row_indices = np.where(solution_list > 0)[0]
        matrix_builder.add_data_lists(mapping[row_indices], [curr_item] * len(row_indices), solution_list[row_indices])

    def build_similarity_matrix(self, df_responses, agg_strategy, filter_sample_method, mapping_matrix):
        """
        It builds the similarity matrix by using a dataframe with all the samples collected from the solver in the
        fit function.

        The samples obtained from the solver are post-processed with a filtering operation (i.e. filter_strategy) and
        an aggregation operation (i.e. agg_strategy). At the end of this pipeline, it outputs a single list containing
        a column of the similarity matrix.

        :param df_responses: a dataframe containing the samples collected from the solver
        :param agg_strategy: the post-processing aggregation to be used on the samples
        :param filter_sample_method: the filter technique used before the post-processing aggregation
        :param mapping_matrix: list of np.ndarray containing the mapping of the samples variables into the original
                               variables for each item problem
        :return: the similarity matrix built from the dataframe given
        """
        n_items = self.URM_train.shape[1]
        matrix_builder = IncrementalSparseMatrix(n_rows=n_items, n_cols=n_items)

        for currentItem in range(n_items):
            response_df = df_responses[df_responses.item_id == currentItem].copy()
            self.add_sample_responses_to_matrix_builder(matrix_builder, agg_strategy, filter_sample_method, response_df,
                                                        currentItem, mapping_matrix[currentItem])

        return sps.csr_matrix(matrix_builder.get_SparseMatrix())

    def fit(self, agg_strategy="FIRST", filter_sample_method="NONE", topK=5, alpha_multiplier=0,
            constraint_multiplier=1, chain_multiplier=1, filter_items_method="NONE", filter_items_n=100,
            num_reads=100, **filter_items_parameters):
        """
        It fits the data (i.e. URM_train) by solving an optimization problem for each item. Each optimization problem
        is generated from the URM_train without the target column and the target column by means of transformation to
        a QUBO based on "transform_fn" with some regulators; then it is solved by a solver given at the initialization
        of the class.

        Then by using the samples collected from the solver, it builds the item-similarity matrix.

        :param agg_strategy: the post-processing aggregation to be used on the samples
        :param filter_sample_method: the filter technique used before the post-processing aggregation
        :param topK: a regulator number that indicates the number of selected variables forced during the optimization
        :param alpha_multiplier: a multiplier number applied on the constraint of the sparsity regulator term
        :param constraint_multiplier: a multiplier number applied on the constraint strength of the variable
                                      selection regulator
        :param chain_multiplier: a multiplier number applied on the chain strength of the embedding
        :param filter_items_method: name of the filtering method to select a set of items for the resolution of the
                                    optimization problem
        :param filter_items_n: number of items to be selected by the filtering method
        :param num_reads: number of samples to compute from the solver
        :param filter_items_parameters: other parameters regarding the filter items method
        """
        self._check_fit_parameters(agg_strategy, filter_items_method, filter_sample_method)
        if filter_items_method == "COSINE":
            self.FILTER_ITEMS_METHODS["COSINE"] = ItemSelectorByCosineSimilarity(**filter_items_parameters)
        URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)

        n_items = URM_train.shape[1]
        item_pop = np.array((URM_train > 0).sum(axis=0)).flatten()

        # Need a labeling of variables to order the variables from 0 to n_items. With variable leading zeros based on
        # the highest number of digits
        leading_zeros = len(str(n_items - 1))
        variables = ["a{:0{}d}".format(i, leading_zeros) for i in range(n_items)]

        if self.to_resume:
            start_item = self.df_responses[self.ITEM_ID_COLUMN_NAME].max()
        else:
            self.df_responses = pd.DataFrame()
            start_item = 0

        self.FILTER_ITEMS_METHODS[filter_items_method].precompute_best_item_indices(URM_train)
        matrix_builder = IncrementalSparseMatrix(n_rows=n_items, n_cols=n_items)

        for curr_item in tqdm(range(start_item, n_items), desc="%s: Computing W_sparse matrix" % self.RECOMMENDER_NAME):
            # get the target column
            target_column = URM_train[:, curr_item].toarray()

            # set the "curr_item"-th column of URM_train to zero
            start_pos = URM_train.indptr[curr_item]
            end_pos = URM_train.indptr[curr_item + 1]
            current_item_data_backup = URM_train.data[start_pos: end_pos].copy()
            URM_train.data[start_pos: end_pos] = 0.0

            # select items to be used in the QUBO optimization problem
            URM = URM_train.copy()
            URM, mapping_array = self.FILTER_ITEMS_METHODS[filter_items_method].filter_items(URM, target_column,
                                                                                             curr_item,
                                                                                             filter_items_n)
            n_variables = len(mapping_array)

            # get BQM/QUBO problem for the current item
            qubo = self.LOSSES[self.obj_function].get_qubo_problem(URM, target_column)
            qubo = qubo + (np.log1p(item_pop[curr_item]) ** 2 + 1) * alpha_multiplier * (np.max(qubo) - np.min(qubo)) \
                   * np.identity(n_variables)
            if topK > -1:
                constraint_strength = max(self.MIN_CONSTRAINT_STRENGTH,
                                          constraint_multiplier * (np.max(qubo) - np.min(qubo)))
                # avoid using the "combinations" function of dimod in order to speed up the computation
                qubo += -2 * constraint_strength * topK * np.identity(n_variables) + constraint_strength * np.ones(
                    (n_variables, n_variables))

            # Generation of the BQM with qubo in a quicker way checked with some performance measuring. On a test of
            # 2000 n_items, this method is quicker w.r.t. from_numpy_matrix function
            bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
            bqm.add_variables_from(dict(zip(variables, np.diag(qubo))))

            for i in range(n_variables):
                values = np.array(qubo[i, i + 1:]).flatten() + np.array(qubo[i + 1:, i]).flatten()
                keys = [(variables[i], variables[j]) for j in range(i + 1, n_variables)]
                bqm.add_interactions_from(dict(zip(keys, values)))

            self._print("The BQM for item {} is {}".format(curr_item, bqm))

            # solve the problem with the solver
            try:
                if ("child_properties" in self.solver.properties and
                    self.solver.properties["child_properties"]["category"] == "qpu") \
                        or "qpu_properties" in self.solver.properties:
                    chain_strength = max(self.MIN_CONSTRAINT_STRENGTH,
                                         chain_multiplier * (np.max(qubo) - np.min(qubo)))
                    response = self.solver.sample(bqm, chain_strength=chain_strength, num_reads=num_reads)
                    self._print("Break chain percentage of item {} is {}"
                                .format(curr_item, list(response.data(fields=["chain_break_fraction"]))))
                    self._print("Timing of QPU is %s" % response.info["timing"])
                else:
                    response = self.solver.sample(bqm, num_reads=num_reads)

                self._print("The response for item {} is {}".format(curr_item, response.aggregate()))
            except OSError as err:
                traceback.print_exc()
                raise err

            # save response in self.responses
            response_df = response.to_pandas_dataframe()
            response_df[self.ITEM_ID_COLUMN_NAME] = curr_item
            if self.do_save_responses:
                self.df_responses = self.df_responses.append(response_df, ignore_index=True)
                self.mapping_matrix.append(mapping_array)
            else:
                self.df_responses = self.df_responses.reindex(sorted(self.df_responses.columns), axis=1)
                self.add_sample_responses_to_matrix_builder(matrix_builder, agg_strategy, filter_sample_method,
                                                            response_df, curr_item, mapping_array)

            # restore URM_train
            URM_train.data[start_pos:end_pos] = current_item_data_backup

        if self.do_save_responses:
            self.df_responses = self.df_responses.reindex(sorted(self.df_responses.columns), axis=1)
            self.W_sparse = self.build_similarity_matrix(self.df_responses, agg_strategy, filter_sample_method,
                                                         self.mapping_matrix)
        else:
            self.W_sparse = matrix_builder.get_SparseMatrix()
