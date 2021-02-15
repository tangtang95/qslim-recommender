import numpy as np
from scipy import sparse as sps

from src.models.QuantumSLIM.ItemSelectors.ItemSelectorInterface import ItemSelectorInterface


class ItemSelectorAll(ItemSelectorInterface):
    def precompute_best_item_indices(self, URM: sps.csr_matrix):
        pass

    def get_sorted_best_item_indices(self, URM: sps.csr_matrix, target_column: np.ndarray, item_idx: int) -> np.ndarray:
        pass

    def filter_items(self, URM: sps.csr_matrix, target_column: np.ndarray, item_idx: int, n_items: int) -> (sps.csr_matrix, np.ndarray):
        return URM, np.arange(0, URM.shape[1])
