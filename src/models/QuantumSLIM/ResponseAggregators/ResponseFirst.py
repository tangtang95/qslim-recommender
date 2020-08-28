from typing import List

import pandas as pd

from src.models.QuantumSLIM.ResponseAggregators.ResponseAggregateStrategy import ResponseAggregateStrategy


class ResponseFirst(ResponseAggregateStrategy):
    def get_aggregated_response(self, response_df: pd.DataFrame) -> List:
        best_samples = response_df[response_df["energy"] == response_df["energy"].min()]
        var_names = [col for col in best_samples.columns.to_list() if col.startswith("a")]
        return best_samples[var_names].to_numpy()[0]