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
from src.models.QuantumSLIM.Aggregators.AggregatorInterface import AggregatorInterface
from src.models.QuantumSLIM.Filters.FilterStrategy import FilterStrategy
from src.models.QuantumSLIM.Filters.NoFilter import NoFilter
from src.models.QuantumSLIM.Losses.LossInterface import LossInterface
from src.models.QuantumSLIM.Losses.MSELoss import MSELoss


class QuantumSLIM_MSE(BaseItemSimilarityMatrixRecommender):
    """
    It trains a Sparse LInear Methods (SLIM) item similarity model by using a Quantum DWave Machine. The objective
    function are MSE losses and some variants with some other regulators.
    """

    RECOMMENDER_NAME = "QuantumSLIM_MSE"

    MIN_CONSTRAINT_STRENGTH = 10
    ITEM_ID_COLUMN_NAME = "item_id"

    def __init__(self, URM_train, solver: dimod.Sampler, agg_strategy: AggregatorInterface = AggregatorFirst(),
                 transform_fn: LossInterface = MSELoss(only_positive=False),
                 filter_strategy: FilterStrategy = NoFilter(), verbose=True):
        super(QuantumSLIM_MSE, self).__init__(URM_train, verbose=verbose)
        self.solver = solver
        self.agg_strategy = agg_strategy
        self.transform_fn = transform_fn
        self.filter_strategy = filter_strategy
        self.df_responses = None
        self.to_resume = False

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
            response_df = df_responses[df_responses.item_id == currentItem].copy()
            filtered_response_df = self.filter_strategy.filter_samples(response_df)
            solution_list = self.agg_strategy.get_aggregated_response(filtered_response_df)

            self._print("The aggregated response for item {} is {}".format(currentItem,
                                                                           solution_list))

            row_indices = np.where(solution_list > 0)[0]
            matrix_builder.add_data_lists(row_indices, [currentItem] * len(row_indices), solution_list[row_indices])
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

        # Choose unpopular items
        item_pop = np.array((URM_train > 0).sum(axis=0)).flatten()
        unpopular_items_indices = np.where(item_pop < unpopular_threshold)[0]

        # Need a labeling of variables to order the variables from 0 to n_items. With variable leading zeros based on
        # the highest number of digits
        leading_zeros = len(str(n_items - 1))
        variables = ["a{:0{}d}".format(i, leading_zeros) for i in range(n_items)]

        if self.to_resume:
            start_item = self.df_responses[self.ITEM_ID_COLUMN_NAME].max()
        else:
            self.df_responses = pd.DataFrame()
            start_item = 0

        for curr_item in tqdm(range(start_item, n_items), desc="%s: Computing W_sparse matrix" % self.RECOMMENDER_NAME):
            # get the target column
            target_column = URM_train[:, curr_item].toarray()

            # set the "curr_item"-th column of URM_train to zero
            start_pos = URM_train.indptr[curr_item]
            end_pos = URM_train.indptr[curr_item + 1]
            current_item_data_backup = URM_train.data[start_pos: end_pos].copy()
            URM_train.data[start_pos: end_pos] = 0.0

            # get BQM/QUBO problem for the current item
            qubo = self.transform_fn.get_qubo_problem(URM_train, target_column)
            qubo = np.round(qubo * qubo_round_percentage)
            qubo = qubo + (np.log1p(item_pop[curr_item])**2 + 1) * alpha_multiplier * (np.max(qubo) - np.min(qubo)) \
                   * np.identity(n_items)
            if topK > -1:
                constraint_strength = max(self.MIN_CONSTRAINT_STRENGTH,
                                          constraint_multiplier * (np.max(qubo) - np.min(qubo)))
                # avoid using the "combinations" function of dimod in order to speed up the computation
                qubo += -2 * constraint_strength * topK * np.identity(n_items) + constraint_strength * np.ones(
                    (n_items, n_items))

            # Generation of the BQM with qubo in a quicker way checked with some performance measuring. On a test of
            # 2000 n_items, this method has a boost from 22 seconds to 12 seconds for a single item w.r.t.
            # from_numpy_matrix function
            bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
            bqm.add_variables_from(dict(zip(variables, np.diag(qubo))))

            for i in range(n_items):
                values = np.array(qubo[i, i + 1:]).flatten() + np.array(qubo[i + 1:, i]).flatten()
                keys = [(variables[i], variables[j]) for j in range(i + 1, n_items)]
                bqm.add_interactions_from(dict(zip(keys, values)))

            # remove unpopular items from the optimization problem
            bqm.fix_variables({variables[i] for i in unpopular_items_indices})

            self._print("The BQM for item {} is {}".format(curr_item, bqm))

            # solve the problem with the solver
            try:
                if ("child_properties" in self.solver.properties and
                    self.solver.properties["child_properties"]["category"] == "qpu") \
                        or "qpu_properties" in self.solver.properties:
                    chain_strength = max(self.MIN_CONSTRAINT_STRENGTH,
                                         chain_multiplier * (np.max(qubo) - np.min(qubo)))
                    response = self.solver.sample(bqm, chain_strength=chain_strength, **solver_parameters)
                    self._print("Break chain percentage of item {} is {}"
                                .format(curr_item, list(response.data(fields=["chain_break_fraction"]))))
                    self._print("Timing of QPU is %s" % response.info["timing"])
                else:
                    response = self.solver.sample(bqm, **solver_parameters)

                self._print("The response for item {} is {}".format(curr_item, response.aggregate()))
            except OSError as err:
                traceback.print_exc()
                raise err

            # save response in self.responses
            response_df = response.to_pandas_dataframe()
            response_df[variables[curr_item]] = 0.0
            for i in unpopular_items_indices:
                response_df[variables[i]] = 0.0
            response_df[self.ITEM_ID_COLUMN_NAME] = curr_item
            self.df_responses = self.df_responses.append(response_df, ignore_index=True)

            # restore URM_train
            URM_train.data[start_pos:end_pos] = current_item_data_backup

        self.df_responses = self.df_responses.reindex(sorted(self.df_responses.columns), axis=1)
        self.W_sparse = self.build_similarity_matrix(self.df_responses)
