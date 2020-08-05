from abc import ABC, abstractmethod

import dimod


class ResponseAggregateStrategy(ABC):

    @abstractmethod
    def get_aggregated_response(self, response: dimod.SampleSet):
        pass
