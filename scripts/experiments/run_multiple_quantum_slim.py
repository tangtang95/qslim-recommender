import itertools
import os
import traceback
from datetime import datetime

from scripts.experiments.run_quantum_slim import get_arguments, save_result, run_experiment

import argparse
import numpy as np

from src.utils.utilities import get_project_root_path

USE_LIST = False

HYPERPARAMETERS = {
    "loss": ["NORM_MEAN_ERROR", "NORM_MEAN_ERROR_SQUARED"],
    "aggregation": ["FIRST"],
    "top_k": [-1],
    "alpha_mlt": [0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    "constr_mlt": [1.0],
    "chain_mlt": [1.0]
}

# use in case of failed experiments
LIST_HYPERPARAMETERS = []

if __name__ == '__main__':
    arguments = get_arguments()
    dict_args = vars(arguments).copy()
    failed_experiments_hyperparameters = []

    hyperparameters_combinations = list(itertools.product(*HYPERPARAMETERS.values())) \
        if not USE_LIST else LIST_HYPERPARAMETERS
    for i, hyperparameters in enumerate(hyperparameters_combinations):
        print("Experiment n.{}/{}: the hyperparameters are {}:{}".format(i+1, len(hyperparameters_combinations),
                                                                         HYPERPARAMETERS.keys(), list(hyperparameters)))
        for j, key in enumerate(HYPERPARAMETERS.keys()):
            dict_args[key] = hyperparameters[j]

        curr_args = argparse.Namespace(**dict_args)

        try:
            mdl, result = run_experiment(curr_args)
            print("Results: {}".format(str(result)))

            if arguments.save_result:
                save_result(mdl, result, curr_args)
        except:
            traceback.print_exc()
            print("Experiment n.{} with these parameters {}:{} failed!".format(i+1, HYPERPARAMETERS.keys(),
                                                                               hyperparameters))
            failed_experiments_hyperparameters.append(hyperparameters)

    if len(failed_experiments_hyperparameters) > 0:
        failed_exp_filepath = os.path.join(get_project_root_path(), "scripts", "experiments",
                                           "failed_experiments_list.txt")
        with open(failed_exp_filepath, 'w') as f:
            datestring = datetime.now().strftime('%b%d_%H-%M-%S')
            f.write("File generated at {}\n".format(datestring))
            f.write("Failed experiments with the hyperparameters {} are: \n{}\n".format(HYPERPARAMETERS.keys(),
                                                                                  failed_experiments_hyperparameters))
