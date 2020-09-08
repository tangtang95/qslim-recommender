import numpy as np
import pandas as pd

from src.models.QuantumSLIM.Aggregators.AggregatorInterface import AggregatorInterface


class AggregatorUnion(AggregatorInterface):
    def __init__(self, operator_fn: callable, is_filter_first: bool, is_weighted: bool):
        self.operator_fn = operator_fn
        self.is_filter_first = is_filter_first
        self.is_weighted = is_weighted

    def get_aggregated_response(self, response_df: pd.DataFrame) -> np.ndarray:
        best_samples = response_df[response_df["energy"] == response_df["energy"].min()]
        var_names = [col for col in best_samples.columns.to_list() if col.startswith("a")]
        first_sample = best_samples[var_names].to_numpy()[0]

        if self.is_weighted:
            response_df["weight"] = - response_df["energy"]

            if (response_df["weight"].max() - response_df["weight"].min()) != 0:
                response_df["weight"] = (response_df["weight"] - response_df["weight"].min()) / \
                                        (response_df["weight"].max() - response_df["weight"].min())
            else:
                response_df["weight"] = 1

            response_df["num_occurrences"] = response_df["num_occurrences"] * response_df["weight"]

        response_df[var_names] = response_df[var_names] * \
                                      response_df["num_occurrences"].to_numpy().reshape([-1, 1])
        agg_series = response_df.aggregate(sum)
        aggregation = agg_series[var_names].to_numpy()
        aggregation = self.operator_fn(aggregation)
        aggregation = aggregation / agg_series["num_occurrences"]

        if self.is_filter_first:
            aggregation[first_sample != 1] = 0

        return aggregation
