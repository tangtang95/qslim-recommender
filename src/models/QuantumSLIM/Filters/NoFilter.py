import pandas as pd

from src.models.QuantumSLIM.Filters.FilterStrategy import FilterStrategy


class NoFilter(FilterStrategy):

    def filter_samples(self, response_df: pd.DataFrame):
        return response_df.copy()
