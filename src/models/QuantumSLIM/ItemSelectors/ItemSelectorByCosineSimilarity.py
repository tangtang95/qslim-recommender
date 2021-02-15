import numpy as np
from scipy import sparse as sps

from course_lib.Base.Similarity.Compute_Similarity import Compute_Similarity
from src.models.QuantumSLIM.ItemSelectors.ItemSelectorInterface import ItemSelectorInterface


class ItemSelectorByCosineSimilarity(ItemSelectorInterface):

    def __init__(self, topK=10000, shrink=0, normalize=False):
        super().__init__()
        self.topK = topK
        self.shrink = shrink
        self.normalize = normalize

    def precompute_best_item_indices(self, URM: sps.csr_matrix):
        similarity = Compute_Similarity(URM, shrink=self.shrink, topK=self.topK, normalize=self.normalize,
                                        similarity="cosine")
        similarity_matrix = similarity.compute_similarity()
        self.sorted_indices = np.array(np.argsort(-similarity_matrix.todense(), axis=1))

    def get_sorted_best_item_indices(self, URM: sps.csr_matrix, target_column: np.ndarray, item_idx: int) -> np.ndarray:
        return self.sorted_indices[item_idx]
