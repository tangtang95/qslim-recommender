import argparse
import itertools
import os
import time
import pandas as pd

import scipy.sparse as sps

from scripts.experiments.run_quantum_slim import get_solver, get_loss, get_aggregation_strategy, get_filter_strategy, \
    get_arguments
from src.models.QuantumSLIM.QSLIM_Timing import QSLIM_Timing
from src.utils.utilities import handle_folder_creation, get_project_root_path

N_EXPERIMENTS = 5

HYPERPARAMETERS = {
    "N_USERS": [10000],
    "N_ITEMS": [750],
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
    _fit_time_start = time.time()
    kwargs = {}
    if args.num_reads > 0:
        kwargs["num_reads"] = args.num_reads
    try:
        model.fit(topK=args.top_k, alpha_multiplier=args.alpha_mlt, constraint_multiplier=args.constr_mlt,
                  chain_multiplier=args.chain_mlt, unpopular_threshold=args.unpop_thresh,
                  qubo_round_percentage=args.round_percent, **kwargs)
    except OSError:
        print("EXCEPTION: handling exception by saving the model up to now in order to resume it later")
        return model, {}
    fit_time = time.time() - _fit_time_start
    # END FIT TIME
    return model, fit_time


def save_result(n_users, n_items, density, fit_times_dict, qpu_times_dict, total_fit_time_list, args):
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
          total_fit_time_list, fit_times_dict['preprocessing_time'], fit_times_dict['sampling_time'],
          fit_times_dict['response_save_time'], fit_times_dict['postprocessing_time'],
          qpu_times_dict['qpu_sampling_time'], qpu_times_dict['qpu_anneal_time_per_sample'],
          qpu_times_dict['qpu_readout_time_per_sample'], qpu_times_dict['qpu_programming_time'],
          qpu_times_dict['qpu_delay_time_per_sample']]], columns=columns)
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

        fit_times_dict = {
            'preprocessing_time': [],
            'sampling_time': [],
            'response_save_time': [],
            'postprocessing_time': []
        }
        qpu_times_dict = {
            'qpu_sampling_time': [],
            'qpu_anneal_time_per_sample': [],
            'qpu_readout_time_per_sample': [],
            'qpu_programming_time': [],
            'qpu_delay_time_per_sample': []
        }
        URM_train = sps.rand(n_users, n_items, density=density)

        total_fit_time_list = []
        for i in range(N_EXPERIMENTS):
            model, fit_time = run_time_test(URM_train, curr_args)
            print("Total fit time is: %s" % fit_time)
            print(model.fit_time)
            print(model.qpu_time)
            for key in fit_times_dict.keys():
                fit_times_dict[key].append(model.fit_time[key])
            for key in qpu_times_dict.keys():
                qpu_times_dict[key].append(model.qpu_time[key])
            total_fit_time_list.append(fit_time)

        save_result(n_users, n_items, density, fit_times_dict, qpu_times_dict, total_fit_time_list, args=curr_args)
