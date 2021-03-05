import pandas as pd
import os
import numpy as np

from src.utils.utilities import get_project_root_path

if __name__ == '__main__':
    #np.random.seed(1234)
    data_path = os.path.join(get_project_root_path(), "data", "ml100k")

    input_filepath = os.path.join(data_path, "ml100k.tsv")
    df = pd.read_csv(filepath_or_buffer=input_filepath, sep="\t", header=None)

    #df[2] = 1
    df = df.drop(columns=[3])

    output_filepath = os.path.join(data_path, "ml100k.csv")
    df.to_csv(output_filepath, header=False, index=False)

    """
    item_chosen_filepath = os.path.join(data_path, "{}_item_chosen.txt".format(prefix_name))
    with open(item_chosen_filepath, 'w') as f:
        for item in item_chosen:
            f.write("%s\n" % item)
    """
