import argparse
import multiprocessing
import os
import numpy as np
from datetime import datetime
from functools import partial

from skopt.space import Integer, Categorical, Real

from course_lib.Base.Evaluation.Evaluator import EvaluatorHoldout
from course_lib.Data_manager.DataSplitter_k_fold import DataSplitter_Warm_k_fold
from course_lib.ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from course_lib.ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from scripts.experiments.run_quantum_slim import get_solver
from src.data.NoHeaderCSVReader import NoHeaderCSVReader
from src.models.QuantumSLIM.QuantumSLIM_MSE import QuantumSLIM_MSE
from src.utils.utilities import get_project_root_path, str2bool

SOLVER_TYPE_LIST = ["QPU", "SA", "HYBRID", "FIXED_QPU", "CLIQUE_FIXED_QPU"]
SOLVER_NAME_LIST = ["DW_2000Q", "ADVANTAGE", "HYBRID_V1", "HYBRID_V2"]
QPU_SOLVER_NAME_LIST = SOLVER_NAME_LIST[:2]
HYBRID_SOLVER_NAME_LIST = SOLVER_NAME_LIST[2:4]
LOSS_NAMES = QuantumSLIM_MSE.get_implemented_losses()

# DATASET DEFAULT VALUES
DEFAULT_N_FOLDS = 5
DEFAULT_N_CASES = 100
DEFAULT_N_RANDOM_STARTS = 10

# CONSTRUCTOR DEFAULT VALUES
DEFAULT_SOLVER_TYPE = "SA"
DEFAULT_SOLVER_NAME = "NONE"
DEFAULT_LOSS = "NORM_MSE"

# FIT DEFAULT VALUES
DEFAULT_NUM_READS = 50
DEFAULT_FILTER_ITEMS_N = 100

# OTHERS
DEFAULT_CUTOFF = 5
DEFAULT_OUTPUT_FOLDER = os.path.join(get_project_root_path(), "report", "quantum_slim")

def get_arguments():
    """
    Defining the arguments available for the script

    :return: argument parser
    """
    parser = argparse.ArgumentParser()

    # Data setting
    parser.add_argument("-f", "--filename", help="File name of the CSV dataset to load (the file has to"
                                                 "be stored in data folder)", required=True, type=str)
    parser.add_argument("-n", "--n_folds", help="Number of holds to split", default=DEFAULT_N_FOLDS, type=int)
    parser.add_argument("-nc", "--n_cases", help="Number of cases to search during the bayesian search",
                        type=int, default=DEFAULT_N_CASES)
    parser.add_argument("-nrs", "--n_random_start", help="Number of random starts done by the bayesian search",
                        type=int, default=DEFAULT_N_RANDOM_STARTS)


    # Quantum SLIM setting
    parser.add_argument("-st", "--solver_type", help="Type of solver used for Quantum SLIM", choices=SOLVER_TYPE_LIST,
                        type=str, default=DEFAULT_SOLVER_TYPE)
    parser.add_argument("-sn", "--solver_name", help="Name of the solver to be used", choices=SOLVER_NAME_LIST,
                        type=str, default=DEFAULT_SOLVER_NAME)
    parser.add_argument("-l", "--loss", help="Loss function to use in Quantum SLIM", choices=LOSS_NAMES,
                        type=str, default=DEFAULT_LOSS)

    # Quantum SLIM Fit setting
    parser.add_argument("-nr", "--num_reads", help="Number of reads to be done for each sample call of the solver",
                        type=int, default=DEFAULT_NUM_READS)
    parser.add_argument("-fin", "--filter_items_n", help="Number of items to be selected by the item selection method",
                        type=int, default=DEFAULT_FILTER_ITEMS_N)

    # Evaluation setting
    parser.add_argument("-c", "--cutoff", help="Cutoff value for evaluation", type=int,
                        default=DEFAULT_CUTOFF)

    # Store results
    parser.add_argument("-o", "--output_folder", default=DEFAULT_OUTPUT_FOLDER,
                        help="Basic folder where to store the output", type=str)

    # Others
    parser.add_argument("-p", "--parallelize", help="Whether to parallelize computation over item selection type",
                        type=str2bool, default=True)
    parser.add_argument("-dis", "--do_item_selection", help="Whether to do item selection over the items",
                        type=str2bool, default=True)
    parser.add_argument("-t", "--token", help="Token string in order to use DWave Sampler", type=str)

    return parser.parse_args()


def run_QSLIM_on_item_selection(item_selection_type, parameterSearch, parameter_search_space, recommender_input_args,
                                n_cases, n_random_starts, resume_from_saved, save_model, output_folder_path,
                                output_file_name_root, metric_to_optimize, recommender_input_args_last_test=None):
    original_parameter_search_space = parameter_search_space

    hyperparameters_range_dictionary = {}
    hyperparameters_range_dictionary["topK"] = Integer(3, 50)
    hyperparameters_range_dictionary["filter_items_method"] = Categorical([item_selection_type])
    hyperparameters_range_dictionary["agg_strategy"] = Categorical(QuantumSLIM_MSE.get_implemented_aggregators())
    #hyperparameters_range_dictionary["filter_sample_method"] = Categorical(QuantumSLIM_MSE.get_implemented_filter_samples_methods())
    hyperparameters_range_dictionary["constraint_multiplier"] = Real(low=0, high=5, prior='uniform')

    if item_selection_type == "COSINE":
        hyperparameters_range_dictionary["shrink"] = Integer(0, 100)
        hyperparameters_range_dictionary["normalize"] = Categorical([1, 0])

    local_parameter_search_space = {**hyperparameters_range_dictionary, **original_parameter_search_space}

    parameterSearch.search(recommender_input_args,
                           parameter_search_space=local_parameter_search_space,
                           n_cases=n_cases,
                           n_random_starts=n_random_starts,
                           resume_from_saved=resume_from_saved,
                           save_model=save_model,
                           output_folder_path=output_folder_path,
                           output_file_name_root=output_file_name_root + "_" + item_selection_type,
                           metric_to_optimize=metric_to_optimize,
                           recommender_input_args_last_test=recommender_input_args_last_test)


def runParameterSearch_QSLIM(URM_train, solver, n_reads=50, filter_items_n=100, URM_train_last_test=None,
                             metric_to_optimize="MAP",
                             evaluator_validation=None, evaluator_test=None,
                             output_folder_path="result_experiments/", parallelizeKNN=True,
                             n_cases=35, n_random_starts=5, resume_from_saved=False, save_model="best",
                             item_selection_list=None):

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    output_file_name_root = QuantumSLIM_MSE.RECOMMENDER_NAME

    URM_train = URM_train.copy()

    if URM_train_last_test is not None:
        URM_train_last_test = URM_train_last_test.copy()

    parameterSearch = SearchBayesianSkopt(QuantumSLIM_MSE, evaluator_validation=evaluator_validation,
                                          evaluator_test=evaluator_test)

    if item_selection_list is None:
        item_selection_list = [x for x in QuantumSLIM_MSE.get_implemented_filter_item_methods() if x != "NONE"]

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, solver, "NORM_MSE"],
        CONSTRUCTOR_KEYWORD_ARGS={"verbose": False},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={"filter_items_n": filter_items_n, "num_reads": n_reads}
    )

    if URM_train_last_test is not None:
        recommender_input_args_last_test = recommender_input_args.copy()
        recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test
    else:
        recommender_input_args_last_test = None

    run_KNNCFRecommender_on_similarity_type_partial = partial(run_QSLIM_on_item_selection,
                                                              recommender_input_args=recommender_input_args,
                                                              parameter_search_space={},
                                                              parameterSearch=parameterSearch,
                                                              n_cases=n_cases,
                                                              n_random_starts=n_random_starts,
                                                              resume_from_saved=resume_from_saved,
                                                              save_model=save_model,
                                                              output_folder_path=output_folder_path,
                                                              output_file_name_root=output_file_name_root,
                                                              metric_to_optimize=metric_to_optimize,
                                                              recommender_input_args_last_test=recommender_input_args_last_test)

    if parallelizeKNN:
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(), maxtasksperchild=1)
        pool.map(run_KNNCFRecommender_on_similarity_type_partial, item_selection_list)
        pool.close()
        pool.join()
    else:
        for similarity_type in item_selection_list:
            run_KNNCFRecommender_on_similarity_type_partial(similarity_type)

    return


def run_experiment(args):
    np.random.seed(52316)

    reader = NoHeaderCSVReader(filename=args.filename)
    splitter = DataSplitter_Warm_k_fold(reader, n_folds=args.n_folds)
    splitter.load_data()

    URM_train, URM_val, URM_test = splitter.get_holdout_split()

    evaluator_val = EvaluatorHoldout(URM_val, cutoff_list=[args.cutoff])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[args.cutoff])

    solver = get_solver(args.solver_type, args.solver_name, args.token)

    date_string = datetime.now().strftime('%b%d_%H-%M-%S/')
    output_folder_path = os.path.join(args.output_folder, date_string)

    item_selection_list = None if args.do_item_selection else ["NONE"]
    item_selection_list = ["COSINE"]

    runParameterSearch_QSLIM(URM_train, solver, n_reads=args.num_reads, filter_items_n=args.filter_items_n,
                             URM_train_last_test=URM_train + URM_val, item_selection_list=item_selection_list,
                             metric_to_optimize="MAP", evaluator_validation=evaluator_val,
                             evaluator_test=evaluator_test, output_folder_path=output_folder_path, n_cases=args.n_cases,
                             n_random_starts=args.n_random_start, resume_from_saved=False, save_model="best",
                             parallelizeKNN=args.parallelize)


if __name__ == '__main__':
    args = get_arguments()
    run_experiment(args)
