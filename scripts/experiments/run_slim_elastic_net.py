import numpy as np

from course_lib.Base.Evaluation.Evaluator import EvaluatorHoldout
from course_lib.Base.NonPersonalizedRecommender import TopPop
from course_lib.Data_manager.DataSplitter_k_fold import DataSplitter_Warm_k_fold
from course_lib.SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from src.data.NoHeaderCSVReader import NoHeaderCSVReader

if __name__ == '__main__':
    np.random.seed(52316)

    reader = NoHeaderCSVReader(filename="jester100_t0.csv")
    splitter = DataSplitter_Warm_k_fold(reader, n_folds=5)
    splitter.load_data()

    URM_train, URM_val, URM_test = splitter.get_holdout_split()

    model = SLIMElasticNetRecommender(URM_train)
    model.fit(topK=10)

    evaluator = EvaluatorHoldout(URM_val, cutoff_list=[5])
    print(evaluator.evaluateRecommender(model))
