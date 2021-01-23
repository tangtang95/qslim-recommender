import argparse
import itertools
import os
import time

import dimod
import pandas as pd
import numpy as np

import scipy.sparse as sps
from tqdm import tqdm

from course_lib.Base.Recommender_utils import check_matrix
from scripts.experiments.run_quantum_slim import get_solver, get_loss, get_aggregation_strategy, get_filter_strategy, \
    get_arguments
from src.models.QuantumSLIM.QSLIM_Timing import QSLIM_Timing
from src.utils.utilities import handle_folder_creation, get_project_root_path

N_EXPERIMENTS = 5

HYPERPARAMETERS = {
    "N_USERS": [1000],
    "N_ITEMS": [600, 800],
    "DENSITY": [0.05]
}


def run_time_test(URM_train, args):
    solver = get_solver(args.solver_type, args.solver_name, args.token)
    loss_fn = get_loss(args.loss)
    agg_strategy = get_aggregation_strategy(args.aggregation)
    filter_strategy = get_filter_strategy(args.filter, args.top_filter)
    model = QSLIM_Timing(URM_train=URM_train, solver=solver, transform_fn=loss_fn, agg_strategy=agg_strategy,
                         filter_strategy=filter_strategy, verbose=args.verbose)

    # START FIT TIME

    unpopular_threshold = args.unpop_thresh

    URM_train = check_matrix(URM_train, 'csc', dtype=np.float32)
    n_items = URM_train.shape[1]
    item_pop = np.array((URM_train > 0).sum(axis=0)).flatten()
    unpopular_items_indices = np.where(item_pop < unpopular_threshold)[0]
    variables = ["a{:04d}".format(x) for x in range(n_items)]

    _preprocessing_time_start = time.time()

    for curr_item in tqdm(range(0, n_items), desc="%s: Computing W_sparse matrix" % QSLIM_Timing.RECOMMENDER_NAME):
        # START COLLECTING PREPROCESSSING TIME
        target_column = URM_train[:, curr_item].toarray()
        start_pos = URM_train.indptr[curr_item]
        end_pos = URM_train.indptr[curr_item + 1]
        current_item_data_backup = URM_train.data[start_pos: end_pos].copy()
        URM_train.data[start_pos: end_pos] = 0.0
        qubo = loss_fn.get_qubo_problem(URM_train, target_column)
        qubo = np.round(qubo * args.round_percent)
        qubo = qubo + (np.log1p(item_pop[curr_item]) ** 2 + 1) * args.alpha_mlt * (np.max(qubo) - np.min(qubo)) \
               * np.identity(n_items)
        if args.top_k > -1:
            constraint_strength = max(QSLIM_Timing.MIN_CONSTRAINT_STRENGTH,
                                      args.constr_mlt * (np.max(qubo) - np.min(qubo)))
            qubo += -2 * constraint_strength * args.top_k * np.identity(n_items) + constraint_strength * np.ones(
                (n_items, n_items))

        bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
        bqm.add_variables_from(dict(zip(variables, np.diag(qubo))))

        for i in range(n_items):
            values = np.array(qubo[i, i + 1:]).flatten() + np.array(qubo[i + 1:, i]).flatten()
            keys = [(variables[i], variables[j]) for j in range(i + 1, n_items)]
            bqm.add_interactions_from(dict(zip(keys, values)))
        bqm.fix_variables({"a{:04d}".format(i): 0 for i in unpopular_items_indices})
        # END COLLECTING PREPROCESSING TIME

    preprocessing_time = time.time() - _preprocessing_time_start
    # END FIT TIME
    return model, preprocessing_time


def save_result(n_users, n_items, density, preprocessing_time_list, args):
    # Set up writing folder and file
    results_path = os.path.join(get_project_root_path(), "report", "quantum_slim_timing_tests")
    fd, folder_path_with_date = handle_folder_creation(result_path=results_path,
                                                       filename="results.txt")

    fd.write("--- Quantum SLIM Experiment ---\n")
    fd.write(" - Number of experiments: {}\n".format(N_EXPERIMENTS))
    fd.write("\n")

    fd.write("DATASET INFO\n")
    fd.write(" - Dataset name: RANDOMLY DISTRIBUTED SYNTHETIC DATASET\n")
    fd.write(" - N_USERS: {}\n".format(n_users))
    fd.write(" - N_ITEMS: {}\n".format(n_items))
    fd.write(" - DENSITY: {}\n".format(density))
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
    fd.write(" - QUBO round percentage: {}\n".format(args.round_percent))
    fd.write("\n")

    fd.write("EVALUATION\n")
    fd.write(" - Cutoff: {}\n".format(args.cutoff))
    fd.close()

    # write data and timing info on csv file
    columns = ['n_users', 'n_items', 'density', 'n_exp', 'solver_type', 'solver_name',
               'loss_func', 'agg_strategy', 'filter_strategy', 'top_filter_value',
               'select_k', 'num_reads', 'alpha_mlt', 'constr_mlt', 'chain_mlt',
               'unpop_thresh', 'qubo_round_perc', 'eval_cutoff', 'total_fit_time',
               'preprocessing_time', 'sampling_time', 'response_save_time',
               'postprocessing_time',
               'qpu_sampling_time', 'qpu_anneal_time_per_sample', 'qpu_readout_time_per_sample',
               'qpu_programming_time', 'qpu_delay_time_per_sample'
               ]
    data_info = pd.DataFrame(
        [[n_users, n_items, density, N_EXPERIMENTS, args.solver_type, args.solver_name, args.loss,
          args.aggregation, args.filter, args.top_filter, args.top_k, args.num_reads, args.alpha_mlt,
          args.constr_mlt, args.chain_mlt, args.unpop_thresh, args.round_percent, args.cutoff,
          0, preprocessing_time_list, 0, 0, 0, 0, 0, 0, 0, 0]], columns=columns)
    data_info.to_csv(os.path.join(folder_path_with_date, "experiment_info.csv"), sep=',', header=True, index=False)

    return folder_path_with_date


if __name__ == '__main__':
    arguments = get_arguments()
    dict_args = vars(arguments).copy()

    hyperparameters_combinations = list(itertools.product(*HYPERPARAMETERS.values()))

    for i, hyperparameters in enumerate(hyperparameters_combinations):
        print("Experiment n.{}/{}: the hyperparameters are {}:{}".format(i + 1, len(hyperparameters_combinations),
                                                                         HYPERPARAMETERS.keys(), list(hyperparameters)))
        n_users, n_items, density = hyperparameters[:3]
        for j, key in enumerate(HYPERPARAMETERS.keys()):
            if j <= 2:
                continue
            dict_args[key] = hyperparameters[j]
        curr_args = argparse.Namespace(**dict_args)

        URM_train = sps.rand(n_users, n_items, density=density)

        preprocessing_time_list = []
        for i in range(N_EXPERIMENTS):
            model, preprocessing_time = run_time_test(URM_train, curr_args)
            print("Total preprocessing time is: %s" % preprocessing_time)
            preprocessing_time_list.append(preprocessing_time)

        save_result(n_users, n_items, density, preprocessing_time_list, args=curr_args)
