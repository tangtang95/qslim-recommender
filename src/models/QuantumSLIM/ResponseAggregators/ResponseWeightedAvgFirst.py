from typing import List

import dimod
import numpy as np

from src.models.QuantumSLIM.ResponseAggregators.ResponseAggregateStrategy import ResponseAggregateStrategy


class ResponseWeightedAvgFirst(ResponseAggregateStrategy):
    def get_aggregated_response(self, response: dimod.SampleSet) -> List:
        aggregation = np.zeros(len(response.variables))
        tot_weight = 0

        sample_weight = np.zeros(len(list(response.data())))
        for i, data in enumerate(response.data()):
            sample_weight[i] = - data.energy
        sample_weight = (sample_weight - np.min(sample_weight)) / (np.max(sample_weight) - np.min(sample_weight))

        for i, data in enumerate(response.data()):
            aggregation += np.array(list(data.sample.values())) * (data.num_occurrences * sample_weight[i])
            tot_weight += data.num_occurrences * sample_weight[i]

        aggregation[np.array(list(response.first.sample.values())) != 1] = 0
        return aggregation / tot_weight
