import argparse
import os

import neal
import numpy as np
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler

from course_lib.Base.Evaluation.Evaluator import EvaluatorHoldout
from course_lib.Data_manager.DataSplitter_k_fold import DataSplitter_Warm_k_fold
from src.data.NoHeaderCSVReader import NoHeaderCSVReader
from src.models.QuantumSLIM.Filters.NoFilter import NoFilter
from src.models.QuantumSLIM.Filters.TopFilter import TopFilter
from src.models.QuantumSLIM.QuantumSLIM_MSE import QuantumSLIM_MSE
from src.models.QuantumSLIM.ResponseAggregators.ResponseFirst import ResponseFirst
from src.models.QuantumSLIM.ResponseAggregators.ResponseGenericOperation import ResponseGenericOperation

from src.models.QuantumSLIM.Transformations.MSETransformation import MSETransformation
from src.models.QuantumSLIM.Transformations.NormMSETransformation import NormMSETransformation
from src.utils.utilities import handle_folder_creation, get_project_root_path

SOLVER_NAMES = ["QPU", "SA"]
LOSS_NAMES = ["MSE", "NORM_MSE", "NON_ZERO_MSE", "NON_ZERO_NORM_MSE"]
AGGREGATION_NAMES = ["FIRST", "LOG", "LOG_FIRST", "EXP", "EXP_FIRST", "AVG", "AVG_FIRST", "WEIGHTED_AVG",
                     "WEIGHTED_AVG_FIRST"]
FILTER_NAMES = ["NONE", "TOP"]

DEFAULT_N_FOLDS = 5
DEFAULT_SOLVER = "SA"
DEFAULT_LOSS = "MSE"
DEFAULT_AGGREGATION = "FIRST"
DEFAULT_FILTER = "NONE"
DEFAULT_FILTER_TOP_VALUE = 0.2
DEFAULT_TOP_K = 5
DEFAULT_NUM_READS = 50
DEFAULT_MULTIPLIER = 1.0
DEFAULT_CUTOFF = 5
DEFAULT_OUTPUT_FOLDER = os.path.join(get_project_root_path(), "report", "quantum_slim")

DWAVE_TOKEN = 'DEV-0303e4137a04f495870145d26be7d5b735899b07'


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

    # Quantum SLIM setting
    parser.add_argument("-s", "--solver", help="Solver used for Quantum SLIM", choices=SOLVER_NAMES, type=str,
                        default=DEFAULT_SOLVER)
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
    parser.add_argument("-nr", "--num_reads", help="Number of reads to be done for each sample call of the solver",
                        type=int, default=DEFAULT_NUM_READS)
    parser.add_argument("-com", "--constr_mlt", help="Constraint multiplier of the QUBO that fixes the selection"
                                                     "of variables to k",
                        type=float, default=DEFAULT_MULTIPLIER)
    parser.add_argument("-chm", "--chain_mlt", help="Chain multiplier of the auto-embedding component",
                        type=float, default=DEFAULT_MULTIPLIER)

    # Evaluation setting
    parser.add_argument("-c", "--cutoff", help="Cutoff value for evaluation", type=int,
                        default=DEFAULT_CUTOFF)

    # Store results
    parser.add_argument("-sr", "--save_result", help="Whether to store results or not", type=lambda x: int(x) != 0,
                        default=1)
    parser.add_argument("-o", "--output_folder", default=DEFAULT_OUTPUT_FOLDER,
                        help="Basic folder where to store the output", type=str)

    return parser.parse_args()


def get_solver(solver_name):
    if solver_name == "SA":
        solver = neal.SimulatedAnnealingSampler()
    elif solver_name == "QPU":
        solver = DWaveSampler(token=DWAVE_TOKEN)
        solver = EmbeddingComposite(solver)
    else:
        raise NotImplementedError("Solver {} is not implemented".format(solver_name))
    return solver


def get_loss(loss_name):
    if loss_name == "MSE":
        loss_fn = MSETransformation(only_positive=False)
    elif loss_name == "NORM_MSE":
        loss_fn = NormMSETransformation(only_positive=False)
    elif loss_name == "NON_ZERO_MSE":
        loss_fn = MSETransformation(only_positive=True)
    elif loss_name == "NON_ZERO_NORM_MSE":
        loss_fn = NormMSETransformation(only_positive=True)
    else:
        raise NotImplementedError("Loss function {} is not implemented".format(loss_name))
    return loss_fn


def get_aggregation_strategy(aggregation_name):
    log_operation_fn = lambda arr: np.log1p(arr)
    no_operation_fn = lambda arr: arr
    exp_operation_fn = lambda arr: np.exp(arr)

    if aggregation_name == "FIRST":
        agg_strategy = ResponseFirst()
    elif aggregation_name == "LOG":
        agg_strategy = ResponseGenericOperation(log_operation_fn, is_filter_first=False, is_weighted=False)
    elif aggregation_name == "LOG_FIRST":
        agg_strategy = ResponseGenericOperation(log_operation_fn, is_filter_first=True, is_weighted=False)
    elif aggregation_name == "EXP":
        agg_strategy = ResponseGenericOperation(exp_operation_fn, is_filter_first=False, is_weighted=False)
    elif aggregation_name == "EXP_FIRST":
        agg_strategy = ResponseGenericOperation(exp_operation_fn, is_filter_first=True, is_weighted=False)
    elif aggregation_name == "AVG":
        agg_strategy = ResponseGenericOperation(no_operation_fn, is_filter_first=False, is_weighted=False)
    elif aggregation_name == "AVG_FIRST":
        agg_strategy = ResponseGenericOperation(no_operation_fn, is_filter_first=True, is_weighted=False)
    elif aggregation_name == "WEIGHTED_AVG":
        agg_strategy = ResponseGenericOperation(no_operation_fn, is_filter_first=False, is_weighted=True)
    elif aggregation_name == "WEIGHTED_AVG_FIRST":
        agg_strategy = ResponseGenericOperation(no_operation_fn, is_filter_first=True, is_weighted=True)
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


def run_experiment(args):
    reader = NoHeaderCSVReader(filename=args.filename)
    splitter = DataSplitter_Warm_k_fold(reader, n_folds=args.n_folds)
    splitter.load_data()

    URM_train, URM_val, URM_test = splitter.get_holdout_split()

    solver = get_solver(args.solver)
    loss_fn = get_loss(args.loss)
    agg_strategy = get_aggregation_strategy(args.aggregation)
    filter_strategy = get_filter_strategy(args.filter, args.top_filter)
    model = QuantumSLIM_MSE(URM_train=URM_train, solver=solver, transform_fn=loss_fn, agg_strategy=agg_strategy,
                            filter_strategy=filter_strategy)
    model.fit(topK=args.top_k, num_reads=args.num_reads, constraint_multiplier=args.constr_mlt,
              chain_multiplier=args.chain_mlt)

    evaluator = EvaluatorHoldout(URM_val, cutoff_list=[args.cutoff])
    return model, evaluator.evaluateRecommender(model)[0]


if __name__ == '__main__':
    arguments = get_arguments()
    model, result = run_experiment(arguments)
    print("Results: {}".format(str(result)))

    if arguments.save_result:
        # Set up writing folder and file
        fd, folder_path_with_date = handle_folder_creation(result_path=arguments.output_folder)

        # Save model
        model.save_model(folder_path=folder_path_with_date)
        model.df_responses.to_csv(os.path.join(folder_path_with_date, "solver_responses.csv"), index=False)

        fd.write("--- Quantum SLIM Experiment ---\n")
        fd.write("\n")

        fd.write("CONSTRUCTOR PARAMETERS\n")
        fd.write(" - Solver: {}\n".format(arguments.solver))
        fd.write(" - Loss function: {}\n".format(arguments.loss))
        fd.write(" - Aggregation strategy: {}\n".format(arguments.aggregation))
        fd.write(" - Filter strategy: {}\n".format(arguments.filter))
        fd.write(" - Top filter value: {}\n".format(arguments.top_filter))
        fd.write("\n")

        fd.write("FIT PARAMETERS\n")
        fd.write(" - Top K: {}\n".format(arguments.top_k))
        fd.write(" - Number of reads: {}\n".format(arguments.num_reads))
        fd.write(" - Constraint multiplier: {}\n".format(arguments.constr_mlt))
        fd.write(" - Chain multiplier: {}\n".format(arguments.chain_mlt))
        fd.write("\n")

        fd.write("- Results -\n")
        fd.write(str(result))

        fd.close()

