from abc import ABC, abstractmethod

import pandas as pd
import numpy as np


class AggregatorInterface(ABC):

    @abstractmethod
    def get_aggregated_response(self, response_df: pd.DataFrame) -> np.ndarray:
        pass
