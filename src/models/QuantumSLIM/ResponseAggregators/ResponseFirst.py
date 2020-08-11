from typing import List

import dimod

from src.models.QuantumSLIM.ResponseAggregators.ResponseAggregateStrategy import ResponseAggregateStrategy


class ResponseFirst(ResponseAggregateStrategy):
    def get_aggregated_response(self, response: dimod.SampleSet) -> List:
        return list(response.first.sample.values())
