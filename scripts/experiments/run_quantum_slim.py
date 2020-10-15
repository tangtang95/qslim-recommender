import argparse
import os

import neal
import numpy as np
import pandas as pd
from dwave.system import LazyFixedEmbeddingComposite
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler, DWaveCliqueSampler, LeapHybridSampler

from course_lib.Base.Evaluation.Evaluator import EvaluatorHoldout
from course_lib.Data_manager.DataSplitter_k_fold import DataSplitter_Warm_k_fold
from src.data.NoHeaderCSVReader import NoHeaderCSVReader
from src.models.QuantumSLIM.Aggregators.AggregatorFirst import AggregatorFirst
from src.models.QuantumSLIM.Aggregators.AggregatorUnion import AggregatorUnion
from src.models.QuantumSLIM.Filters.NoFilter import NoFilter
from src.models.QuantumSLIM.Filters.TopFilter import TopFilter
from src.models.QuantumSLIM.Losses.MSELoss import MSELoss
from src.models.QuantumSLIM.Losses.NormMSELoss import NormMSELoss
from src.models.QuantumSLIM.Losses.NormMeanErrorLoss import NormMeanErrorLoss
from src.models.QuantumSLIM.QuantumSLIM_MSE import QuantumSLIM_MSE
from src.utils.utilities import handle_folder_creation, get_project_root_path, str2bool

SOLVER_TYPE_LIST = ["QPU", "SA", "HYBRID", "FIXED_QPU", "CLIQUE_FIXED_QPU"]
SOLVER_NAME_LIST = ["DW_2000Q", "ADVANTAGE", "HYBRID_V1", "HYBRID_V2"]
QPU_SOLVER_NAME_LIST = SOLVER_NAME_LIST[:2]
HYBRID_SOLVER_NAME_LIST = SOLVER_NAME_LIST[2:4]
LOSS_NAMES = ["MSE", "NORM_MSE", "NON_ZERO_MSE", "NON_ZERO_NORM_MSE", "SIM_NORM_MSE", "SIM_NON_ZERO_NORM_MSE",
              "NORM_MEAN_ERROR", "NORM_MEAN_ERROR_SQUARED"]
AGGREGATION_NAMES = ["FIRST", "LOG", "LOG_FIRST", "EXP", "EXP_FIRST", "AVG", "AVG_FIRST", "WEIGHTED_AVG",
                     "WEIGHTED_AVG_FIRST"]
FILTER_NAMES = ["NONE", "TOP"]

# DATASET DEFAULT VALUES
DEFAULT_N_FOLDS = 5

# CONSTRUCTOR DEFAULT VALUES
DEFAULT_SOLVER_TYPE = "SA"
DEFAULT_SOLVER_NAME = "NONE"
DEFAULT_LOSS = "NORM_MSE"
DEFAULT_AGGREGATION = "FIRST"
DEFAULT_FILTER = "NONE"
DEFAULT_FILTER_TOP_VALUE = 0.2

# FIT DEFAULT VALUES
DEFAULT_TOP_K = 5
DEFAULT_ALPHA_MULTIPLIER = 0.0
DEFAULT_NUM_READS = 50
DEFAULT_MULTIPLIER = 1.0
DEFAULT_UNPOPULAR_THRESHOLD = 0

# OTHERS
DEFAULT_CUTOFF = 5
DEFAULT_OUTPUT_FOLDER = os.path.join(get_project_root_path(), "report", "quantum_slim")
DEFAULT_RESPONSES_CSV_FILENAME = "solver_responses.csv"


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

    # Cache setting
    parser.add_argument("-fp", "--foldername", help="Folder name of the folder in report/quantum_slim containing "
                                                    "samples responses from a"
                                                    "previous experiment. If this is compiled, then the solver is not"
                                                    "used", type=str)

    # Quantum SLIM setting
    parser.add_argument("-st", "--solver_type", help="Type of solver used for Quantum SLIM", choices=SOLVER_TYPE_LIST,
                        type=str, default=DEFAULT_SOLVER_TYPE)
    parser.add_argument("-sn", "--solver_name", help="Name of the solver to be used", choices=SOLVER_NAME_LIST,
                        type=str, default=DEFAULT_SOLVER_NAME)
    parser.add_argument("-l", "--loss", help="Loss function to use in Quantum SLIM", choices=LOSS_NAMES,
                        type=str, default=DEFAULT_LOSS)
    parser.add_argument("-g", "--aggregation", help="Type of aggregation to use on the response of Quantum SLIM solver",
                        choices=AGGREGATION_NAMES, type=str, default=DEFAULT_AGGREGATION)
    parser.add_argument("-fi", "--filter", help="Type of filtering to use on the response of Quantum SLIM solver",
                        choices=FILTER_NAMES, type=str, default=DEFAULT_FILTER)
    parser.add_argument("-tfi", "--top_filter", help="Percentage of top filtering to use on the response "
                                                     "of Quantum SLIM solver",
                        type=float, default=DEFAULT_FILTER_TOP_VALUE)

    # Quantum SLIM Fit setting
    parser.add_argument("-k", "--top_k", help="Number of similar item selected for each item", type=int,
                        default=DEFAULT_TOP_K)
    parser.add_argument("-am", "--alpha_mlt", help="Alpha multiplier of the linear sparsity regulator term", type=float,
                        default=DEFAULT_ALPHA_MULTIPLIER)
    parser.add_argument("-nr", "--num_reads", help="Number of reads to be done for each sample call of the solver",
                        type=int, default=DEFAULT_NUM_READS)
    parser.add_argument("-com", "--constr_mlt", help="Constraint multiplier of the QUBO that fixes the selection"
                                                     "of variables to k",
                        type=float, default=DEFAULT_MULTIPLIER)
    parser.add_argument("-chm", "--chain_mlt", help="Chain multiplier of the auto-embedding component",
                        type=float, default=DEFAULT_MULTIPLIER)
    parser.add_argument("-ut", "--unpop_thresh", help="The threshold of unpopularity for the removal of unpopular "
                                                      "items",
                        type=int, default=DEFAULT_UNPOPULAR_THRESHOLD)

    # Evaluation setting
    parser.add_argument("-c", "--cutoff", help="Cutoff value for evaluation", type=int,
                        default=DEFAULT_CUTOFF)

    # Store results
    parser.add_argument("-v", "--verbose", help="Whether to output on command line much as possible",
                        type=str2bool, default=True)
    parser.add_argument("-sr", "--save_result", help="Whether to store results or not", type=str2bool, default=True)
    parser.add_argument("-o", "--output_folder", default=DEFAULT_OUTPUT_FOLDER,
                        help="Basic folder where to store the output", type=str)

    # Others
    parser.add_argument("-t", "--token", help="Token string in order to use DWave Sampler", type=str)

    return parser.parse_args()


def get_solver(solver_type: str, solver_name: str, token):
    filters = {}
    if solver_type.endswith("QPU") and solver_name in QPU_SOLVER_NAME_LIST:
        filters["topology__type"] = "pegasus" if solver_name == "ADVANTAGE" else "chimera"
        filters["name__contains"] = "Advantage_system" if solver_name == "ADVANTAGE" else "DW_2000Q"
    if solver_type.endswith("HYBRID") and solver_name in HYBRID_SOLVER_NAME_LIST:
        filters["name__contains"] = "version2" if solver_name == "HYBRID_V2" else "v1"

    if solver_type == "SA":
        solver = neal.SimulatedAnnealingSampler()
    elif solver_type == "QPU":
        solver = DWaveSampler(client="qpu", solver=filters, token=token)
        solver = EmbeddingComposite(solver)
    elif solver_type == "HYBRID":
        solver = LeapHybridSampler(solver=filters, token=token)
    elif solver_type == "FIXED_QPU":
        solver = DWaveSampler(client="qpu", solver=filters, token=token)
        solver = LazyFixedEmbeddingComposite(solver)
    elif solver_type == "CLIQUE_FIXED_QPU":
        solver = DWaveCliqueSampler(client="qpu", solver=filters, token=token)
    else:
        raise NotImplementedError("Solver {} is not implemented".format(solver_type))
    return solver


def get_loss(loss_name):
    if loss_name == "MSE":
        loss_fn = MSELoss(only_positive=False)
    elif loss_name == "NORM_MSE":
        loss_fn = NormMSELoss(only_positive=False, is_simplified=False)
    elif loss_name == "NON_ZERO_MSE":
        loss_fn = MSELoss(only_positive=True)
    elif loss_name == "NON_ZERO_NORM_MSE":
        loss_fn = NormMSELoss(only_positive=True, is_simplified=False)
    elif loss_name == "SIM_NORM_MSE":
        loss_fn = NormMSELoss(only_positive=False, is_simplified=True)
    elif loss_name == "SIM_NON_ZERO_NORM_MSE":
        loss_fn = NormMSELoss(only_positive=True, is_simplified=True)
    elif loss_name == "NORM_MEAN_ERROR":
        loss_fn = NormMeanErrorLoss(only_positive=False, is_squared=False)
    elif loss_name == "NORM_MEAN_ERROR_SQUARED":
        loss_fn = NormMeanErrorLoss(only_positive=False, is_squared=True)
    else:
        raise NotImplementedError("Loss function {} is not implemented".format(loss_name))
    return loss_fn


def get_aggregation_strategy(aggregation_name):
    log_operation_fn = lambda arr: np.log1p(arr)
    no_operation_fn = lambda arr: arr
    exp_operation_fn = lambda arr: np.exp(arr)

    if aggregation_name == "FIRST":
        agg_strategy = AggregatorFirst()
    elif aggregation_name == "LOG":
        agg_strategy = AggregatorUnion(log_operation_fn, is_filter_first=False, is_weighted=False)
    elif aggregation_name == "LOG_FIRST":
        agg_strategy = AggregatorUnion(log_operation_fn, is_filter_first=True, is_weighted=False)
    elif aggregation_name == "EXP":
        agg_strategy = AggregatorUnion(exp_operation_fn, is_filter_first=False, is_weighted=False)
    elif aggregation_name == "EXP_FIRST":
        agg_strategy = AggregatorUnion(exp_operation_fn, is_filter_first=True, is_weighted=False)
    elif aggregation_name == "AVG":
        agg_strategy = AggregatorUnion(no_operation_fn, is_filter_first=False, is_weighted=False)
    elif aggregation_name == "AVG_FIRST":
        agg_strategy = AggregatorUnion(no_operation_fn, is_filter_first=True, is_weighted=False)
    elif aggregation_name == "WEIGHTED_AVG":
        agg_strategy = AggregatorUnion(no_operation_fn, is_filter_first=False, is_weighted=True)
    elif aggregation_name == "WEIGHTED_AVG_FIRST":
        agg_strategy = AggregatorUnion(no_operation_fn, is_filter_first=True, is_weighted=True)
    else:
        raise NotImplementedError("Aggregation strategy {} is not implemented".format(aggregation_name))
    return agg_strategy


def get_filter_strategy(filter_name, top_filter_value):
    if filter_name == "NONE":
        filter_strategy = NoFilter()
    elif filter_name == "TOP":
        filter_strategy = TopFilter(top_p=top_filter_value)
    else:
        raise NotImplementedError("Filter strategy {} is not implemented".format(filter_name))
    return filter_strategy


def run_experiment(args, preload_df_responses=None):
    np.random.seed(52316)

    reader = NoHeaderCSVReader(filename=args.filename)
    splitter = DataSplitter_Warm_k_fold(reader, n_folds=args.n_folds)
    splitter.load_data()

    URM_train, URM_val, URM_test = splitter.get_holdout_split()

    solver = get_solver(args.solver_type, args.solver_name, args.token)
    loss_fn = get_loss(args.loss)
    agg_strategy = get_aggregation_strategy(args.aggregation)
    filter_strategy = get_filter_strategy(args.filter, args.top_filter)
    model = QuantumSLIM_MSE(URM_train=URM_train, solver=solver, transform_fn=loss_fn, agg_strategy=agg_strategy,
                            filter_strategy=filter_strategy, verbose=args.verbose)
    if preload_df_responses is not None:
        model.preload_fit(preload_df_responses)

    if args.foldername is None:
        kwargs = {}
        if args.num_reads > 0:
            kwargs["num_reads"] = args.num_reads
        try:
            model.fit(topK=args.top_k, alpha_multiplier=args.alpha_mlt, constraint_multiplier=args.constr_mlt,
                      chain_multiplier=args.chain_mlt, unpopular_threshold=args.unpop_thresh, **kwargs)
        except OSError:
            print("EXCEPTION: handling exception by saving the model up to now in order to resume it later")
            return model, {}
    else:
        responses_df = pd.read_csv(os.path.join(args.output_folder, args.foldername, DEFAULT_RESPONSES_CSV_FILENAME))
        model.W_sparse = model.build_similarity_matrix(df_responses=responses_df)

    evaluator = EvaluatorHoldout(URM_val, cutoff_list=[args.cutoff])
    return model, evaluator.evaluateRecommender(model)[0]


def parse_results_file(filepath, keep_pars=None):
    args_dict = {}

    args_match_names = {
        "Dataset name": "filename",
        "N folds split": "n_folds",

        "Solver: ": "solver_type",
        "Solver name": "solver_name",
        "Loss": "loss",
        "Aggregation": "aggregation",
        "Filter": "filter",
        "Top filter": "top_filter",

        "Top K": "top_k",
        "Number of reads": "num_reads",
        "Constraint": "constr_mlt",
        "Alpha": "alpha_mlt",
        "Chain": "chain_mlt",
        "Unpopular threshold": "unpop_thresh",

        "Cutoff": "cutoff",
    }

    if keep_pars is not None:
        args_match_names = {key: elem for key, elem in args_match_names.items() if elem in keep_pars}

    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            for string, parameter in args_match_names.items():
                if line.find(string) != -1:
                    args_dict[parameter] = line.split(": ")[-1]
    return args_dict


def save_result(model, exp_result, args):
    # Set up writing folder and file
    fd, folder_path_with_date = handle_folder_creation(result_path=args.output_folder,
                                                       filename="results.txt" if exp_result != {} else "results_fail.txt")

    # Save model
    if exp_result != {}:
        model.save_model(folder_path=folder_path_with_date)

    if args.foldername is None:
        model.df_responses.to_csv(os.path.join(folder_path_with_date, DEFAULT_RESPONSES_CSV_FILENAME), index=False)
    else:
        cache_results_filepath = os.path.join(args.output_folder, args.foldername, "results.txt")
        parameters_to_overwrite = ["solver_type", "solver_name", "loss", "top_k", "num_reads", "constr_mlt",
                                   "chain_mlt", "alpha_mlt", "unpop_thresh"]
        args_dict = parse_results_file(cache_results_filepath, parameters_to_overwrite)

        # Remove keys to overwrite from original arguments
        init_args_dict = vars(args)
        for parameter in parameters_to_overwrite:
            init_args_dict.pop(parameter)

        args = argparse.Namespace(**{**args_dict, **init_args_dict})

    fd.write("--- Quantum SLIM Experiment ---\n")
    fd.write("\n")

    fd.write("DATASET INFO\n")
    fd.write("Dataset name: {}\n".format(args.filename))
    fd.write("N folds split: {}\n".format(args.n_folds))
    fd.write("\n")

    fd.write("CONSTRUCTOR PARAMETERS\n")
    fd.write(" - Solver: {}\n".format(args.solver_type))
    fd.write(" - Solver name: {}\n".format(args.solver_name))
    fd.write(" - Loss function: {}\n".format(args.loss))
    fd.write(" - Aggregation strategy: {}\n".format(args.aggregation))
    fd.write(" - Filter strategy: {}\n".format(args.filter))
    fd.write(" - Top filter value: {}\n".format(args.top_filter))
    fd.write("\n")

    fd.write("FIT PARAMETERS\n")
    fd.write(" - Top K: {}\n".format(args.top_k))
    fd.write(" - Number of reads: {}\n".format(args.num_reads))
    fd.write(" - Alpha multiplier: {}\n".format(args.alpha_mlt))
    fd.write(" - Constraint multiplier: {}\n".format(args.constr_mlt))
    fd.write(" - Chain multiplier: {}\n".format(args.chain_mlt))
    fd.write(" - Unpopular threshold: {}\n".format(args.unpop_thresh))
    fd.write("\n")

    fd.write("EVALUATION\n")
    fd.write(" - Cutoff: {}\n".format(args.cutoff))

    fd.write("- Results -\n")
    if args.foldername is not None:
        fd.write("The following results comes from the solver_responses.csv of folder {}\n"
                 .format(args.foldername))
    fd.write(str(exp_result))

    fd.close()
    return folder_path_with_date


if __name__ == '__main__':
    arguments = get_arguments()
    mdl, result = run_experiment(arguments)
    print("Results: {}".format(str(result)))

    if arguments.save_result:
        save_result(mdl, result, arguments)


