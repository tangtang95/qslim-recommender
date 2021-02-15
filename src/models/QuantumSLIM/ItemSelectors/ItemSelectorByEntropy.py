import numpy as np
from scipy import sparse as sps
from scipy.stats import entropy

from src.models.QuantumSLIM.ItemSelectors.ItemSelectorInterface import ItemSelectorInterface


class ItemSelectorByEntropy(ItemSelectorInterface):

    def precompute_best_item_indices(self, URM: sps.csr_matrix):
        entropies = entropy(URM.toarray(), axis=0)
        self.sorted_indices = np.argsort(entropies)[::-1]

    def get_sorted_best_item_indices(self, URM: sps.csr_matrix, target_column: np.ndarray, item_idx: int) -> np.ndarray:
        if self.sorted_indices is None:
            entropies = entropy(URM.toarray(), axis=0)
            sorted_indices = np.argsort(entropies)[::-1]
            return sorted_indices
        return self.sorted_indices
