import numpy as np
from scipy import sparse as sps

from src.models.QuantumSLIM.ItemSelectors.ItemSelectorInterface import ItemSelectorInterface


class ItemSelectorByPopularity(ItemSelectorInterface):
    def precompute_best_item_indices(self, URM):
        item_pop = np.array((URM > 0).sum(axis=0)).flatten()
        self.sorted_indices = np.argsort(item_pop)[::-1]

    def get_sorted_best_item_indices(self, URM: sps.csr_matrix, target_column: np.ndarray) -> np.ndarray:
        if self.sorted_indices is None:
            item_pop = np.array((URM > 0).sum(axis=0)).flatten()
            sorted_indices = np.argsort(item_pop)[::-1]
            return sorted_indices
        return self.sorted_indices
