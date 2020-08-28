from abc import ABC, abstractmethod
from typing import List

import pandas as pd


class ResponseAggregateStrategy(ABC):

    @abstractmethod
    def get_aggregated_response(self, response_df: pd.DataFrame) -> List:
        pass
