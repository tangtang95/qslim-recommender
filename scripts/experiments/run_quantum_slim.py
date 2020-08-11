import argparse
import os

import neal
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler

from course_lib.Base.Evaluation.Evaluator import EvaluatorHoldout
from course_lib.Data_manager.DataSplitter_k_fold import DataSplitter_Warm_k_fold
from src.data.NoHeaderCSVReader import NoHeaderCSVReader
from src.models.QuantumSLIM.QuantumSLIM_MSE import QuantumSLIM_MSE
from src.models.QuantumSLIM.ResponseAggregators.ResponseFirst import ResponseFirst
from src.models.QuantumSLIM.ResponseAggregators.ResponseLog import ResponseLog
from src.models.QuantumSLIM.Transformations.MSETransformation import MSETransformation
from src.models.QuantumSLIM.Transformations.NormMSETransformation import NormMSETransformation
from src.utils.utilities import handle_folder_creation, get_project_root_path

SOLVER_NAMES = ["QPU", "SA"]
LOSS_NAMES = ["MSE", "NORM_MSE"]
AGGREGATION_NAMES = ["FIRST", "LOG"]

DEFAULT_N_FOLDS = 5
DEFAULT_SOLVER = "SA"
DEFAULT_LOSS = "MSE"
DEFAULT_AGGREGATION = "FIRST"
DEFAULT_TOP_K = 5
DEFAULT_NUM_READS = 50
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

    # Quantum SLIM Fit setting
    parser.add_argument("-k", "--top_k", help="Number of similar item selected for each item", type=int,
                        default=DEFAULT_TOP_K)
    parser.add_argument("-nr", "--num_reads", help="Number of reads to be done for each sample call of the solver",
                        type=int, default=DEFAULT_NUM_READS)

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
        loss_fn = MSETransformation()
    elif loss_name == "NORM_MSE":
        loss_fn = NormMSETransformation()
    else:
        raise NotImplementedError("Loss function {} is not implemented".format(loss_name))
    return loss_fn


def get_aggregation_strategy(aggregation_name):
    if aggregation_name == "FIRST":
        agg_strategy = ResponseFirst()
    elif aggregation_name == "LOG":
        agg_strategy = ResponseLog()
    else:
        raise NotImplementedError("Aggregation strategy {} is not implemented".format(aggregation_name))
    return agg_strategy


def run_experiment(args):
    reader = NoHeaderCSVReader(filename=args.filename)
    splitter = DataSplitter_Warm_k_fold(reader, n_folds=args.n_folds)
    splitter.load_data()

    URM_train, URM_val, URM_test = splitter.get_holdout_split()

    solver = get_solver(args.solver)
    loss_fn = get_loss(args.loss)
    agg_strategy = get_aggregation_strategy(args.aggregation)
    model = QuantumSLIM_MSE(URM_train=URM_train, solver=solver, transform_fn=loss_fn, agg_strategy=agg_strategy)
    model.fit(topK=args.top_k, num_reads=args.num_reads)

    evaluator = EvaluatorHoldout(URM_val, cutoff_list=[args.cutoff])
    return evaluator.evaluateRecommender(model)


if __name__ == '__main__':
    arguments = get_arguments()
    result = run_experiment(arguments)
    print("Results: {}".format(str(result)))

    if arguments.save_result:
        # Set up writing folder and file
        fd, folder_path_with_date = handle_folder_creation(result_path=arguments.output_folder)

        fd.write("--- Quantum SLIM Experiment ---\n")
        fd.write("\n")

        fd.write("CONSTRUCTOR PARAMETERS\n")
        fd.write(" - Solver: {}\n".format(arguments.solver))
        fd.write(" - Loss function: {}\n".format(arguments.loss))
        fd.write(" - Aggregation strategy: {}\n".format(arguments.aggregation))
        fd.write("\n")

        fd.write("FIT PARAMETERS\n")
        fd.write(" - Top K: {}\n".format(arguments.top_k))
        fd.write(" - Number of reads: {}\n".format(arguments.num_reads))
        fd.write("\n")

        fd.write("- Results -\n")
        fd.write(str(result))

        fd.close()
