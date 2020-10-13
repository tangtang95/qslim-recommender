import pandas as pd
import os
import numpy as np

from src.utils.utilities import get_project_root_path

if __name__ == '__main__':
    np.random.seed(1234)
    data_path = os.path.join(get_project_root_path(), "data", "ml100k")

    input_filepath = os.path.join(data_path, "ml100k.tsv")
    df = pd.read_csv(filepath_or_buffer=input_filepath, sep="\t", header=None)

    item_indices, counts = np.unique(df[1], return_counts=True)
    pop_item_indices = item_indices[counts > 5]
    item_chosen = np.random.choice(pop_item_indices, 100, replace=False)
    small_df = df[df[1].isin(item_chosen)]

    _, item_pop = np.unique(small_df[1], return_counts=True)
    print("Item popularity of chosen items are: {}".format(item_pop))

    prefix_name = "medium"
    output_filepath = os.path.join(data_path, "{}_ml100k.csv".format(prefix_name))
    small_df.to_csv(output_filepath, header=False, index=False)

    item_chosen_filepath = os.path.join(data_path, "{}_item_chosen.txt".format(prefix_name))
    with open(item_chosen_filepath, 'w') as f:
        for item in item_chosen:
            f.write("%s\n" % item)
