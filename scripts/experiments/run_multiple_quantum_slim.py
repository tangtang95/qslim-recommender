import itertools

from scripts.experiments.run_quantum_slim import get_arguments, save_result, run_experiment

import argparse

USE_LIST = False

HYPERPARAMETERS = {
    "top_k": [5, 10, 20],
    "aggregation": ["LOG", "LOG_FIRST", "AVG", "AVG_FIRST"]
}

# use in case of failed experiments
LIST_HYPERPARAMETERS = []

if __name__ == '__main__':
    arguments = get_arguments()
    dict_args = vars(arguments).copy()

    hyperparameters_combinations = list(itertools.product(*HYPERPARAMETERS.values())) \
        if not USE_LIST else LIST_HYPERPARAMETERS
    for i, hyperparameters in enumerate(hyperparameters_combinations):
        print("Experiment n.{}/{}: the hyperparameters are {}:{}".format(i+1, len(hyperparameters_combinations),
                                                                         HYPERPARAMETERS.keys(), list(hyperparameters)))
        for j, key in enumerate(HYPERPARAMETERS.keys()):
            dict_args[key] = hyperparameters[j]

        curr_args = argparse.Namespace(**dict_args)

        mdl, result = run_experiment(curr_args, do_fit=arguments.foldername is None)
        print("Results: {}".format(str(result)))

        if arguments.save_result:
            save_result(mdl, result, curr_args)
