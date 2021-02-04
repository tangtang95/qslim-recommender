import numpy as np
from scipy import sparse as sps

from src.models.QuantumSLIM.ItemSelectors.ItemSelectorInterface import ItemSelectorInterface


class ItemSelectorByVariance(ItemSelectorInterface):

    def precompute_best_item_indices(self, URM: sps.csr_matrix):
        c_URM = URM.copy()
        c_URM.data **= 2
        variances = np.array(c_URM.mean(axis=0) - np.power(URM.mean(axis=0), 2)).flatten()
        self.sorted_indices = np.argsort(variances)[::-1]

    def get_sorted_best_item_indices(self, URM: sps.csr_matrix, target_column: np.ndarray) -> np.ndarray:
        if self.sorted_indices is None:
            c_URM = URM.copy()
            c_URM.data **= 2
            variances = np.array(c_URM.mean(axis=0) - np.power(URM.mean(axis=0), 2)).flatten()
            sorted_indices = np.argsort(variances)[::-1]
            return sorted_indices
        return self.sorted_indices
