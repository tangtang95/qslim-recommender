from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class AggregatorInterface(ABC):

    @abstractmethod
    def get_aggregated_response(self, response_df: pd.DataFrame) -> np.ndarray:
        pass
