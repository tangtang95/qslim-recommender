import os
import pandas as pd

from course_lib.Data_manager.DataReader import DataReader
from course_lib.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs
from src.utils.utilities import get_project_root_path


def load_URM(file_path, separator):
    URM_all_builder = IncrementalSparseMatrix_FilterIDs()

    df_original = pd.read_csv(filepath_or_buffer=file_path, sep=separator, header=None,
                              usecols=[0, 1, 2],
                              dtype={0: str, 1: str, 2: int})

    user_id_list = df_original[0].values
    item_id_list = df_original[1].values
    rating_list = df_original[2].values

    URM_all_builder.add_data_lists(user_id_list, item_id_list, rating_list)
    return URM_all_builder.get_SparseMatrix(), \
           URM_all_builder.get_column_token_to_id_mapper(), \
           URM_all_builder.get_row_token_to_id_mapper()


class NoHeaderCSVReader(DataReader):
    AVAILABLE_URM = ["URM_all"]

    IS_IMPLICIT = False

    def __init__(self, filename, reload_from_original=False):
        super().__init__(reload_from_original_data=reload_from_original)
        root_path = os.path.join(get_project_root_path(), "data")
        self.URM_path = os.path.join(root_path, filename)
        self.DATASET_SUBFOLDER = "{}/".format(filename.split(".")[0])

        self._LOADED_UCM_DICT = {}
        self._LOADED_UCM_MAPPER_DICT = {}

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER

    def _load_from_original_file(self):
        # Load data from original file

        print("RecSys2019Reader: Loading original data")

        URM_all, self.item_original_ID_to_index, self.user_original_ID_to_index = load_URM(self.URM_path,
                                                                                           separator=',')
        self._LOADED_URM_DICT["URM_all"] = URM_all
        self._LOADED_GLOBAL_MAPPER_DICT["user_original_ID_to_index"] = self.user_original_ID_to_index
        self._LOADED_GLOBAL_MAPPER_DICT["item_original_ID_to_index"] = self.item_original_ID_to_index

        print("RecSys2019Reader: loading complete")
