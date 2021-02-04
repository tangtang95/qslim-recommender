import os
from datetime import datetime

import pandas as pd

from src.utils.utilities import get_project_root_path

REPORT_ROOT_PATH = os.path.join(get_project_root_path(), "report", "quantum_slim_timing_tests")
FINAL_REPORT_PATH = os.path.join(get_project_root_path(), "report", "quantum_slim_final_report", "quantum_slim_timing")

if __name__ == '__main__':
    report_files = []
    for root, dirs, files in os.walk(REPORT_ROOT_PATH):
        for file in files:
            if file.endswith(".csv"):
                report_files.append(os.path.join(root, file))
    report_files = sorted(report_files, key=lambda x: os.path.getmtime(x))
    print(report_files)

    df = pd.DataFrame()

    for path_file in report_files:
        df_file = pd.read_csv(path_file, sep=",")
        df = df.append(df_file, ignore_index=True)
    print(df.head())

    date_string = datetime.now().strftime('%b%d_%H-%M-%S')
    df.to_csv(os.path.join(FINAL_REPORT_PATH, "{}.csv".format(date_string)), index=False)
