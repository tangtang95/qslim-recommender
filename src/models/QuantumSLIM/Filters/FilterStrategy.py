from abc import ABC, abstractmethod

import pandas as pd


class FilterStrategy(ABC):

    @abstractmethod
    def filter_samples(self, response_df: pd.DataFrame):
        pass
