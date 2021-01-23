import itertools
import os
import time
import pandas as pd

import scipy.sparse as sps

from course_lib.SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from src.utils.utilities import handle_folder_creation, get_project_root_path

N_EXPERIMENTS = 5

HYPERPARAMETERS = {
    "N_USERS": [1000],
    "N_ITEMS": [100, 200, 300, 400, 500],
    "DENSITY": [0.05]
}


def run_time_test(URM_train):
    model = SLIMElasticNetRecommender(URM_train)

    # START FIT TIME
    _fit_time_start = time.time()

    model.fit(topK=5)

    fit_time = time.time() - _fit_time_start
    # END FIT TIME
    return fit_time


def save_result(n_users, n_items, density, total_fit_time_list):
    # Set up writing folder and file
    results_path = os.path.join(get_project_root_path(), "report", "slim_elastic_net_timing_tests")
    fd, folder_path_with_date = handle_folder_creation(result_path=results_path,
                                                       filename="results.txt")

    fd.write("--- SLIM Elastic Net Experiment ---\n")
    fd.write(" - Number of experiments: {}\n".format(N_EXPERIMENTS))
    fd.write("\n")

    fd.write("DATASET INFO\n")
    fd.write(" - Dataset name: RANDOMLY DISTRIBUTED SYNTHETIC DATASET\n")
    fd.write(" - N_USERS: {}\n".format(n_users))
    fd.write(" - N_ITEMS: {}\n".format(n_items))
    fd.write(" - DENSITY: {}\n".format(density))
    fd.write("\n")

    fd.close()

    # write data and timing info on csv file
    columns = ['n_users', 'n_items', 'density', 'n_exp', 'total_fit_time']
    data_info = pd.DataFrame(
        [[n_users, n_items, density, N_EXPERIMENTS, total_fit_time_list]], columns=columns)
    data_info.to_csv(os.path.join(folder_path_with_date, "experiment_info.csv"), sep=',', header=True, index=False)

    return folder_path_with_date


if __name__ == '__main__':
    hyperparameters_combinations = list(itertools.product(*HYPERPARAMETERS.values()))

    for i, hyperparameters in enumerate(hyperparameters_combinations):
        print("Experiment n.{}/{}: the hyperparameters are {}:{}".format(i + 1, len(hyperparameters_combinations),
                                                                         HYPERPARAMETERS.keys(), list(hyperparameters)))
        n_users, n_items, density = hyperparameters[:3]

        URM_train = sps.rand(n_users, n_items, density=density)

        total_fit_time_list = []
        for i in range(N_EXPERIMENTS):
            fit_time = run_time_test(URM_train)
            print("Total fit time is: %s" % fit_time)
            total_fit_time_list.append(fit_time)

        save_result(n_users, n_items, density, total_fit_time_list)
