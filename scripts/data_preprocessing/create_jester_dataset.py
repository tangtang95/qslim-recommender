import os
import pandas as pd
import numpy as np

from src.utils.utilities import get_project_root_path

if __name__ == '__main__':
    data_path = os.path.join(get_project_root_path(), "data", "jester100")
    input_filepath_list = [os.path.join(data_path, "jester-data-{}.xls".format(i+1)) for i in range(3)]

    df = pd.DataFrame(columns=["user", "item", "rating"])
    last_user_id = 0
    for input_filepath in input_filepath_list:
        dataset = pd.read_excel(input_filepath, header=None)
        dataset = dataset.drop(columns=0)

        mask_df = np.logical_not((dataset == 99) | (dataset <= 0))
        counts = np.sum(mask_df.to_numpy(), axis=1)

        data_matrix = dataset[mask_df].to_numpy()
        index_matrix = np.repeat(np.arange(0, 100).reshape(1, -1), repeats=dataset.shape[0], axis=0)

        items = index_matrix[mask_df.to_numpy()]
        data = np.ones(shape=len(data_matrix[mask_df.to_numpy()]), dtype=int)
        users = np.repeat(np.arange(last_user_id, last_user_id + dataset.shape[0]), repeats=counts)

        df = df.append(pd.DataFrame(data={"user": users, "item": items, "rating": data}))

        last_user_id = last_user_id + dataset.shape[0]

    print(df.head())
    df.to_csv(os.path.join(data_path, "jester100_t0.csv"), sep=",", header=False, index=False)
