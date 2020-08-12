import json
import os
from datetime import datetime

import pandas as pd

from src.utils.utilities import get_project_root_path

REPORT_ROOT_PATH = os.path.join(get_project_root_path(), "report", "quantum_slim")
FINAL_REPORT_PATH = os.path.join(get_project_root_path(), "report", "final_report", "quantum_slim")

if __name__ == '__main__':
    report_files = []
    for root, dirs, files in os.walk(REPORT_ROOT_PATH):
        for file in files:
            if file.endswith(".txt"):
                report_files.append(os.path.join(root, file))

    print(report_files)

    solver_values = []
    loss_values = []

    df = pd.DataFrame(columns=["Datetime", "Algorithm", "Solver", "Loss", "AggregationStrategy", "TopK", "NumReads",
                               "ROC_AUC", "PRECISION", "RECALL", "MAP", "RMSE"])

    for filepath in report_files:
        parameter_values = {}

        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip()
                if line.find("Solver") != -1:
                    parameter_values["Solver"] = line.split(": ")[-1]
                elif line.find("Loss") != -1:
                    parameter_values["Loss"] = line.split(": ")[-1]
                elif line.find("Aggregation") != -1:
                    parameter_values["AggregationStrategy"] = line.split(": ")[-1]
                elif line.find("Top K") != -1:
                    parameter_values["TopK"] = line.split(": ")[-1]
                elif line.find("Number of reads") != -1:
                    parameter_values["NumReads"] = line.split(": ")[-1]
                elif line.find("ROC_AUC") != -1:
                    results = json.loads(line[3:-1].replace("\'", "\""))

        df = df.append({
            "Datetime": filepath.split("/")[-2],
            "Algorithm": "Quantum SLIM",
            "Solver": parameter_values["Solver"],
            "Loss": parameter_values["Loss"],
            "AggregationStrategy": parameter_values["AggregationStrategy"],
            "TopK": parameter_values["TopK"],
            "NumReads": parameter_values["NumReads"],
            "ROC_AUC": results["ROC_AUC"],
            "PRECISION": results["PRECISION"],
            "RECALL": results["RECALL"],
            "MAP": results["MAP"],
            "RMSE": results["RMSE"],
        }, ignore_index=True)
    print(df.head())

    date_string = datetime.now().strftime('%b%d_%H-%M-%S')
    df.to_csv(os.path.join(FINAL_REPORT_PATH, "{}.csv".format(date_string)), index=False)


