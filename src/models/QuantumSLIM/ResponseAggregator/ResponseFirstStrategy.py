from typing import List

import dimod

from src.models.QuantumSLIM.ResponseAggregator.ResponseAggregateStrategy import ResponseAggregateStrategy


class ResponseFirstStrategy(ResponseAggregateStrategy):
    def get_aggregated_response(self, response: dimod.SampleSet) -> List:
        return list(response.first.sample.values())
