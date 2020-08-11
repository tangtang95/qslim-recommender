from course_lib.Base.Evaluation.Evaluator import EvaluatorHoldout
from course_lib.Data_manager.DataSplitter_k_fold import DataSplitter_Warm_k_fold
from course_lib.SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from src.data.NoHeaderCSVReader import NoHeaderCSVReader

if __name__ == '__main__':
    reader = NoHeaderCSVReader(filename="small_ml100k.csv")
    splitter = DataSplitter_Warm_k_fold(reader, n_folds=5)
    splitter.load_data()

    URM_train, URM_val, URM_test = splitter.get_holdout_split()

    model = SLIMElasticNetRecommender(URM_train)
    model.fit(topK=5)

    print(model.W_sparse)

    evaluator = EvaluatorHoldout(URM_val, cutoff_list=[5])
    print(evaluator.evaluateRecommender(model))
