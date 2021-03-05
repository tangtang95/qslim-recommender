import os
import traceback
from datetime import datetime

import numpy as np
from skopt.space import Integer, Real

from course_lib.Base.Evaluation.Evaluator import EvaluatorHoldout
from course_lib.Data_manager.DataSplitter_k_fold import DataSplitter_Warm_k_fold
from course_lib.ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from course_lib.ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from course_lib.SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from scripts.experiments.run_parameter_search_qslim import get_arguments
from src.data.NoHeaderCSVReader import NoHeaderCSVReader
from src.utils.utilities import get_project_root_path

OUTPUT_FOLDER = os.path.join(get_project_root_path(), "report", "slim_elastic_net")


def runParameterSearch_SLIM(URM_train, URM_train_last_test=None,
                            metric_to_optimize="PRECISION",
                            evaluator_validation=None, evaluator_test=None,
                            output_folder_path="result_experiments/",
                            n_cases=35, n_random_starts=5, resume_from_saved=False, save_model="best"):
    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    URM_train = URM_train.copy()

    if URM_train_last_test is not None:
        URM_train_last_test = URM_train_last_test.copy()

    try:

        output_file_name_root = SLIMElasticNetRecommender.RECOMMENDER_NAME

        parameterSearch = SearchBayesianSkopt(SLIMElasticNetRecommender, evaluator_validation=evaluator_validation,
                                              evaluator_test=evaluator_test)

        hyperparameters_range_dictionary = {}
        hyperparameters_range_dictionary["topK"] = Integer(3, 1000)
        hyperparameters_range_dictionary["l1_ratio"] = Real(low=1e-5, high=1.0, prior='log-uniform')
        hyperparameters_range_dictionary["alpha"] = Real(low=1e-3, high=1.0, prior='uniform')

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
            CONSTRUCTOR_KEYWORD_ARGS={},
            FIT_POSITIONAL_ARGS=[],
            FIT_KEYWORD_ARGS={}
        )

        if URM_train_last_test is not None:
            recommender_input_args_last_test = recommender_input_args.copy()
            recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test
        else:
            recommender_input_args_last_test = None

        ## Final step, after the hyperparameter range has been defined for each type of algorithm
        parameterSearch.search(recommender_input_args,
                               parameter_search_space=hyperparameters_range_dictionary,
                               n_cases=n_cases,
                               n_random_starts=n_random_starts,
                               resume_from_saved=resume_from_saved,
                               save_model=save_model,
                               output_folder_path=output_folder_path,
                               output_file_name_root=output_file_name_root,
                               metric_to_optimize=metric_to_optimize,
                               recommender_input_args_last_test=recommender_input_args_last_test)




    except Exception as e:

        print("On recommender {} Exception {}".format(SLIMElasticNetRecommender, str(e)))
        traceback.print_exc()

        error_file = open(output_folder_path + "ErrorLog.txt", "a")
        error_file.write("On recommender {} Exception {}\n".format(SLIMElasticNetRecommender, str(e)))
        error_file.close()


def run_experiment(args):
    np.random.seed(52316)

    reader = NoHeaderCSVReader(filename=args.filename)
    splitter = DataSplitter_Warm_k_fold(reader, n_folds=args.n_folds)
    splitter.load_data()

    URM_train, URM_val, URM_test = splitter.get_holdout_split()

    evaluator_val = EvaluatorHoldout(URM_val, cutoff_list=[args.cutoff])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[args.cutoff])

    date_string = datetime.now().strftime('%b%d_%H-%M-%S/')
    output_folder_path = os.path.join(OUTPUT_FOLDER, date_string)

    runParameterSearch_SLIM(URM_train,
                            URM_train_last_test=URM_train + URM_val,
                            metric_to_optimize="MAP", evaluator_validation=evaluator_val,
                            evaluator_test=evaluator_test, output_folder_path=output_folder_path,
                            n_cases=args.n_cases,
                            n_random_starts=args.n_random_start, resume_from_saved=False, save_model="best")


if __name__ == '__main__':
    args = get_arguments()
    run_experiment(args)
