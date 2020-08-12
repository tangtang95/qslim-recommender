from typing import List

import dimod
import numpy as np

from src.models.QuantumSLIM.ResponseAggregators.ResponseAggregateStrategy import ResponseAggregateStrategy


class ResponseAvg(ResponseAggregateStrategy):
    def get_aggregated_response(self, response: dimod.SampleSet) -> List:
        aggregation = np.zeros(len(response.variables))
        tot_occurrences = 0
        for data in response.data():
            aggregation += np.array(list(data.sample.values())) * data.num_occurrences
            tot_occurrences += data.num_occurrences
        return aggregation / tot_occurrences
