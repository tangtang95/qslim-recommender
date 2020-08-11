import pandas as pd
import os
import numpy as np

from src.utils.utilities import get_project_root_path

if __name__ == '__main__':
    np.random.seed(1234)
    data_path = os.path.join(get_project_root_path(), "data")

    input_filepath = os.path.join(data_path, "ml100k.tsv")
    df = pd.read_csv(filepath_or_buffer=input_filepath, sep="\t", header=None)

    item_indices = df[1].unique()
    item_chosen = np.random.choice(item_indices, 50, replace=False)
    small_df = df[df[1].isin(item_chosen)]

    output_filepath = os.path.join(data_path, "small_ml100k.csv")
    small_df.to_csv(output_filepath, header=False, index=False)

    item_chosen_filepath = os.path.join(data_path, "item_chosen.txt")
    with open(item_chosen_filepath, 'w') as f:
        for item in item_chosen:
            f.write("%s\n" % item)
