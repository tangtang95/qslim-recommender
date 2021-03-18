import numpy as np
from scipy import sparse as sps

from course_lib.Base.IR_feature_weighting import okapi_BM_25, TF_IDF
from course_lib.Base.Recommender_utils import check_matrix
from course_lib.Base.Similarity.Compute_Similarity import Compute_Similarity
from src.models.QuantumSLIM.ItemSelectors.ItemSelectorInterface import ItemSelectorInterface


class ItemSelectorByCosineSimilarity(ItemSelectorInterface):
    """
    Item selector that for each item, it selects a different set of similar items to be kept. The similar items to
    be kept are selected by cosine similarity.
    """
    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]

    def __init__(self, topK=10000, shrink=0, feature_weighting="none", normalize=False):
        super().__init__()
        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError(
                "Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(
                    self.FEATURE_WEIGHTING_VALUES, feature_weighting))

        self.topK = topK
        self.shrink = shrink
        self.normalize = normalize
        self.feature_weighting = feature_weighting

    def precompute_best_item_indices(self, URM: sps.csr_matrix):
        URM = URM.copy()
        if self.feature_weighting == "BM25":
            URM = URM.astype(np.float32)
            URM = okapi_BM_25(URM)
            URM = check_matrix(URM, 'csr')

        elif self.feature_weighting == "TF-IDF":
            URM = URM.astype(np.float32)
            URM = TF_IDF(URM)
            URM = check_matrix(URM, 'csr')

        similarity = Compute_Similarity(URM, shrink=self.shrink, topK=self.topK, normalize=self.normalize,
                                        similarity="cosine")
        similarity_matrix = similarity.compute_similarity()
        self.sorted_indices = np.array(np.argsort(-similarity_matrix.todense(), axis=1))

    def get_sorted_best_item_indices(self, URM: sps.csr_matrix, target_column: np.ndarray, item_idx: int) -> np.ndarray:
        return self.sorted_indices[item_idx]
