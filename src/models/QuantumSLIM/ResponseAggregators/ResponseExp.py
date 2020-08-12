from typing import List

import dimod
import numpy as np

from src.models.QuantumSLIM.ResponseAggregators.ResponseAggregateStrategy import ResponseAggregateStrategy


class ResponseExp(ResponseAggregateStrategy):
    def get_aggregated_response(self, response: dimod.SampleSet) -> List:
        aggregation = np.zeros(len(response.variables))
        for data in response.data():
            aggregation += np.array(list(data.sample.values())) * data.num_occurrences
        aggregation = np.exp(aggregation)
        return (aggregation - aggregation.min()) / (aggregation.max() - aggregation.min())
