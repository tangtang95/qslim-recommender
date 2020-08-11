from abc import ABC, abstractmethod
from typing import List

import dimod


class ResponseAggregateStrategy(ABC):

    @abstractmethod
    def get_aggregated_response(self, response: dimod.SampleSet) -> List:
        pass
