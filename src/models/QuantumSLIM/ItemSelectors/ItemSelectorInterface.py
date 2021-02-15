from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sps


class ItemSelectorInterface(ABC):

    def __init__(self):
        self.sorted_indices = None

    @abstractmethod
    def precompute_best_item_indices(self, URM: sps.csr_matrix):
        pass

    @abstractmethod
    def get_sorted_best_item_indices(self, URM: sps.csr_matrix, target_column: np.ndarray, item_idx: int) -> np.ndarray:
        pass

    def filter_items(self, URM: sps.csr_matrix, target_column: np.ndarray, item_idx: int,
                     n_items: int) -> (sps.csr_matrix, np.ndarray):
        best_indices = self.get_sorted_best_item_indices(URM, target_column, item_idx)[:n_items]
        return URM[:, best_indices], best_indices
