from abc import ABC, abstractmethod
from typing import Tuple, List

import pandas as pd
import numpy as np


class AggregatorInterface(ABC):

    @abstractmethod
    def get_aggregated_response(self, response_df: pd.DataFrame) -> np.ndarray:
        pass
