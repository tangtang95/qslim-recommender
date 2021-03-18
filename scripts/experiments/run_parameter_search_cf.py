from datetime import datetime

import os
import numpy as np

from course_lib.Base.Evaluation.Evaluator import EvaluatorHoldout
from course_lib.Data_manager.DataSplitter_k_fold import DataSplitter_Warm_k_fold
from course_lib.GraphBased.RP3betaRecommender import RP3betaRecommender
from course_lib.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from course_lib.SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from course_lib.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython
from course_lib.ParameterTuning.run_parameter_search import runParameterSearch_Collaborative
from scripts.experiments.run_parameter_search_qslim import get_arguments
from src.data.NoHeaderCSVReader import NoHeaderCSVReader
from src.utils.utilities import get_project_root_path

MODEL = MatrixFactorization_BPR_Cython
OUTPUT_FOLDER = os.path.join(get_project_root_path(), "report", MODEL.RECOMMENDER_NAME)


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

    np.random.seed(None)
    try:
        runParameterSearch_Collaborative(MODEL, URM_train,
                                         URM_train_last_test=URM_train + URM_val,
                                         metric_to_optimize="MAP", evaluator_validation=evaluator_val,
                                         evaluator_test=evaluator_test, output_folder_path=output_folder_path,
                                         n_cases=args.n_cases,
                                         n_random_starts=args.n_random_start, resume_from_saved=False, save_model="best")
    except Exception as e:
        print(e)



if __name__ == '__main__':
    args = get_arguments()
    run_experiment(args)
