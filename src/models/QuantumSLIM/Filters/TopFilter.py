import pandas as pd
import numpy as np

from src.models.QuantumSLIM.Filters.FilterStrategy import FilterStrategy


class TopFilter(FilterStrategy):
    """
    Filter the samples by its energy. The 'top_p'*100 percentage of highest energy samples are kept,
    while the others are dropped.
    """
    def __init__(self, top_p: float):
        self.top_p = top_p

    def filter_samples(self, response_df: pd.DataFrame):
        top_k = int(np.ceil(self.top_p * response_df.shape[0]))
        return response_df.copy().sort_values(by="energy", ascending=True).iloc[:top_k]
