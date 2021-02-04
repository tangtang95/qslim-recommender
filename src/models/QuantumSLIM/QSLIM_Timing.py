import time
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
from src.models.QuantumSLIM.Losses.MSELoss import MSELoss
from src.models.QuantumSLIM.Losses.NormMSELoss import NormMSELoss
from src.models.QuantumSLIM.Losses.NormMeanErrorLoss import NormMeanErrorLoss


class QSLIM_Timing(BaseItemSimilarityMatrixRecommender):
    """
    It trains a Sparse LInear Methods (SLIM) item similarity model by using a Quantum DWave Machine. The objective
    function are MSE losses and some variants with some other regulators.
    """

    RECOMMENDER_NAME = "QSLIM_Timing"

    MIN_CONSTRAINT_STRENGTH = 10

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

    def __init__(self, URM_train, solver: dimod.Sampler, agg_strategy: str = "FIRST", obj_function: str = "NORM_MSE",
                 filter_sample_method: str = "NONE", verbose=True):
        super(QSLIM_Timing, self).__init__(URM_train, verbose=verbose)
        self._check_init_parameters(agg_strategy, obj_function, filter_sample_method)

        self.solver = solver
        self.agg_strategy = agg_strategy
        self.obj_function = obj_function
        self.filter_sample_method = filter_sample_method
        self.df_responses = None
        self.to_resume = False

        # TIMING VARIABLES
        self.fit_time = {
            'preprocessing_time': 0,
            'sampling_time': 0,
            'response_save_time': 0,
            'postprocessing_time': 0
        }
        self.qpu_time = {
            'qpu_sampling_time': 0,
            'qpu_anneal_time_per_sample': 0,
            'qpu_readout_time_per_sample': 0,
            'qpu_programming_time': 0,
            'qpu_delay_time_per_sample': 0
        }

    @classmethod
    def get_implemented_aggregators(cls):
        return list(cls.AGGREGATORS.keys())

    @classmethod
    def get_implemented_losses(cls):
        return list(cls.LOSSES.keys())

    @classmethod
    def get_implemented_filter_samples_methods(cls):
        return list(cls.FILTER_SAMPLES_METHODS.keys())

    def _check_init_parameters(self, aggregation_strategy, obj_function, filter_sample_method):
        if aggregation_strategy not in self.get_implemented_aggregators():
            raise NotImplementedError("Filter strategy {} is not implemented".format(aggregation_strategy))
        if obj_function not in self.get_implemented_losses():
            raise NotImplementedError("Objective function {} is not implemented".format(obj_function))
        if filter_sample_method not in self.get_implemented_filter_samples_methods():
            raise NotImplementedError("Filter sample method {} is not implemented".format(filter_sample_method))

    def _check_fit_parameters(self, filter_item_method):
        if filter_item_method not in self.get_implemented_filter_samples_methods():
            raise NotImplementedError("Filter item method {} is not implemented".format(filter_item_method))

    def preload_fit(self, df_responses):
        """
        Preload the dataframe of df_responses to prepare for the fit function in case you need to resume the fit
        function failed previously.

        :param df_responses: dataframe with all the samples collected by the crashed fit function
        """
        self.df_responses = df_responses
        self.to_resume = True

    def build_similarity_matrix(self, df_responses):
        """
        It builds the similarity matrix by using a dataframe with all the samples collected from the solver in the
        fit function.

        The samples obtained from the solver are post-processed with a filtering operation (i.e. filter_strategy) and
        an aggregation operation (i.e. agg_strategy). At the end of this pipeline, it outputs a single list containing
        a column of the similarity matrix.

        :param df_responses: a dataframe containing the samples collected from the solver
        :return: the similarity matrix built from the dataframe given
        """
        n_items = self.URM_train.shape[1]
        matrix_builder = IncrementalSparseMatrix(n_rows=n_items, n_cols=n_items)

        for currentItem in range(n_items):
            # START COLLECT POSTPROCESSING TIME
            _postprocessing_time_start = time.time()

            response_df = df_responses[df_responses.item_id == currentItem].copy()
            filtered_response_df = self.FILTER_SAMPLES_METHODS[self.filter_sample_method].filter_samples(response_df)
            solution_list = self.AGGREGATORS[self.agg_strategy].get_aggregated_response(filtered_response_df)
            row_indices = np.where(solution_list > 0)[0]
            matrix_builder.add_data_lists(row_indices, [currentItem] * len(row_indices), solution_list[row_indices])

            self.fit_time['postprocessing_time'] += time.time() - _postprocessing_time_start
            # END COLLECTING POSTPROCESSING TIME

        return sps.csr_matrix(matrix_builder.get_SparseMatrix())

    def fit(self, topK=5, alpha_multiplier=0, constraint_multiplier=1, chain_multiplier=1, unpopular_threshold=0,
            qubo_round_percentage=0.01, **solver_parameters):
        """
        It fits the data (i.e. URM_train) by solving an optimization problem for each item. Each optimization problem
        is generated from the URM_train without the target column and the target column by means of transformation to
        a QUBO based on "transform_fn" with some regulators; then it is solved by a solver given at the initialization
        of the class.

        Then by using the samples collected from the solver, it builds the item-similarity matrix.

        :param topK: a regulator number that indicates the number of selected variables forced during the optimization
        :param alpha_multiplier: a multiplier number applied on the constraint of the sparsity regulator term
        :param constraint_multiplier: a multiplier number applied on the constraint strength of the variable
                                      selection regulator
        :param chain_multiplier: a multiplier number applied on the chain strength of the embedding
        :param unpopular_threshold: a number that indicates which are unpopular items (i.e. items with popularity lower
                                    than this threshold are unpopular). These items are removed in the optimization
                                    problem
        :param qubo_round_percentage: a percentage number in [0, 1] that is multiplied to the values of qubo and then
                                      rounded in order to simplify the QPU optimization problem
        :param solver_parameters: other parameters of the sample function of the solver
        """
        URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)
        n_items = URM_train.shape[1]
        item_pop = np.array((URM_train > 0).sum(axis=0)).flatten()
        unpopular_items_indices = np.where(item_pop < unpopular_threshold)[0]
        variables = ["a{:04d}".format(i) for i in range(n_items)]
        if self.to_resume:
            start_item = self.df_responses["item_id"].max()
        else:
            self.df_responses = pd.DataFrame()
            start_item = 0

        for curr_item in tqdm(range(start_item, n_items), desc="%s: Computing W_sparse matrix" % self.RECOMMENDER_NAME):
            # START COLLECTING PREPROCESSSING TIME
            _preprocessing_time_start = time.time()
            target_column = URM_train[:, curr_item].toarray()
            start_pos = URM_train.indptr[curr_item]
            end_pos = URM_train.indptr[curr_item + 1]
            current_item_data_backup = URM_train.data[start_pos: end_pos].copy()
            URM_train.data[start_pos: end_pos] = 0.0
            qubo = self.LOSSES[self.obj_function].get_qubo_problem(URM_train, target_column)
            qubo = np.round(qubo * qubo_round_percentage)
            qubo = qubo + (np.log1p(item_pop[curr_item]) ** 2 + 1) * alpha_multiplier * (np.max(qubo) - np.min(qubo)) \
                   * np.identity(n_items)
            if topK > -1:
                constraint_strength = max(self.MIN_CONSTRAINT_STRENGTH,
                                      constraint_multiplier * (np.max(qubo) - np.min(qubo)))
                qubo += -2 * constraint_strength * topK * np.identity(n_items) + constraint_strength * np.ones(
                    (n_items, n_items))

            bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
            bqm.add_variables_from(dict(zip(variables, np.diag(qubo))))

            for i in range(n_items):
                values = np.array(qubo[i, i + 1:]).flatten() + np.array(qubo[i + 1:, i]).flatten()
                keys = [(variables[i], variables[j]) for j in range(i + 1, n_items)]
                bqm.add_interactions_from(dict(zip(keys, values)))
            bqm.fix_variables({"a{:04d}".format(i): 0 for i in unpopular_items_indices})
            self.fit_time['preprocessing_time'] += (time.time() - _preprocessing_time_start)
            # END COLLECTING PREPROCESSING TIME

            # START COLLECTING SAMPLING TIME
            _sampling_time_start = time.time()
            try:
                if ("child_properties" in self.solver.properties and
                    self.solver.properties["child_properties"]["category"] == "qpu") \
                        or "qpu_properties" in self.solver.properties:
                    chain_strength = max(self.MIN_CONSTRAINT_STRENGTH,
                                         chain_multiplier * (np.max(qubo) - np.min(qubo)))
                    response = self.solver.sample(bqm, chain_strength=chain_strength, **solver_parameters)

                    self._print("Timing of QPU is %s" % response.info["timing"])
                    timing = response.info["timing"]
                    for key in self.qpu_time.keys():
                        self.qpu_time[key] = self.qpu_time[key] + timing[key]
                else:
                    response = self.solver.sample(bqm, **solver_parameters)
            except OSError as err:
                traceback.print_exc()
                raise err

            self.fit_time['sampling_time'] += time.time() - _sampling_time_start
            # END COLLECTING SAMPLING TIME

            # START COLLECTING RESPONSE SAVE TIME
            _response_save_time_start = time.time()

            response_df = response.to_pandas_dataframe()
            response_df["a{:04d}".format(curr_item)] = 0.0
            for i in unpopular_items_indices:
                response_df["a{:04d}".format(i)] = 0.0
            response_df["item_id"] = curr_item
            self.df_responses = self.df_responses.append(response_df, ignore_index=True)
            URM_train.data[start_pos:end_pos] = current_item_data_backup

            self.fit_time['response_save_time'] += time.time() - _response_save_time_start
            # END COLLECTING RESPONSE SAVE TIME

        self.df_responses = self.df_responses.reindex(sorted(self.df_responses.columns), axis=1)
        self.W_sparse = self.build_similarity_matrix(self.df_responses)
