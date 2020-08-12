from typing import List

import numpy as np
import dimod

from src.models.QuantumSLIM.ResponseAggregators.ResponseAggregateStrategy import ResponseAggregateStrategy


class ResponseLogFirst(ResponseAggregateStrategy):
    def get_aggregated_response(self, response: dimod.SampleSet) -> List:
        aggregation = np.zeros(len(response.variables))
        for data in response.data():
            aggregation += np.array(list(data.sample.values())) * data.num_occurrences
        aggregation = np.log(aggregation + 1)
        aggregation = (aggregation - aggregation.min()) / (aggregation.max() - aggregation.min())
        aggregation[np.array(list(response.first.sample.values())) != 1] = 0
        return aggregation
