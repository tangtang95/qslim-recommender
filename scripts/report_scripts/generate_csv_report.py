import json
import os
from datetime import datetime

import pandas as pd
import numpy as np

from src.utils.utilities import get_project_root_path

REPORT_ROOT_PATH = os.path.join(get_project_root_path(), "report", "quantum_slim")
FINAL_REPORT_PATH = os.path.join(get_project_root_path(), "report", "quantum_slim_final_report", "quantum_slim")

if __name__ == '__main__':
    report_files = []
    for root, dirs, files in os.walk(REPORT_ROOT_PATH):
        for file in files:
            if file.endswith(".txt"):
                report_files.append(os.path.join(root, file))

    print(report_files)

    solver_values = []
    loss_values = []

    df = pd.DataFrame(columns=["Datetime", "Algorithm", "SolverType", "SolverName", "Loss", "AggregationStrategy",
                               "FilterStrategy", "TopFilterValue", "TopK", "NumReads",
                               "AlphaMultiplier",
                               "ConstraintMultiplier", "ChainMultiplier", "UnpopularThreshold",
                               "Cached", "PRECISION", "RECALL", "MAP", "NDCG", "AVERAGE_POPULARITY",
                               "DIVERSITY_MEAN_INTER_LIST", "COVERAGE_ITEM", "CHAIN_BREAK_FRACTION_MAX",
                               "CHAIN_BREAK_FRACTION_MEAN", "CHAIN_BREAK_FRACTION_STD"])

    for filepath in report_files:
        parameter_values = {}
        parameter_values["SolverName"] = ""
        parameter_values["ChainMultiplier"] = 1.0
        parameter_values["ConstraintMultiplier"] = 1.0
        parameter_values["AlphaMultiplier"] = 0.0
        parameter_values["RemoveUnpopularity"] = "None"
        parameter_values["UnpopularThreshold"] = 0
        parameter_values["Cached"] = "No"
        parameter_values["CHAIN_BREAK_FRACTION_MAX"] = ""
        parameter_values["CHAIN_BREAK_FRACTION_MEAN"] = ""
        parameter_values["CHAIN_BREAK_FRACTION_STD"] = ""

        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip()
                if line.find("Solver: ") != -1:
                    parameter_values["SolverType"] = line.split(": ")[-1]
                elif line.find("Solver name") != -1:
                    parameter_values["SolverName"] = line.split(": ")[-1]
                elif line.find("Loss") != -1:
                    parameter_values["Loss"] = line.split(": ")[-1]
                elif line.find("Aggregation") != -1:
                    parameter_values["AggregationStrategy"] = line.split(": ")[-1]
                elif line.find("Filter strategy") != -1:
                    parameter_values["FilterStrategy"] = line.split(": ")[-1]
                elif line.find("Top filter value") != -1:
                    parameter_values["TopFilterValue"] = line.split(": ")[-1]
                elif line.find("Top K") != -1:
                    parameter_values["TopK"] = line.split(": ")[-1]
                elif line.find("Number of reads") != -1:
                    parameter_values["NumReads"] = line.split(": ")[-1]
                elif line.find("Constraint") != -1:
                    parameter_values["ConstraintMultiplier"] = line.split(": ")[-1]
                elif line.find("Alpha") != -1:
                    parameter_values["AlphaMultiplier"] = line.split(": ")[-1]
                elif line.find("Chain") != -1:
                    parameter_values["ChainMultiplier"] = line.split(": ")[-1]
                elif line.find("following results") != -1:
                    parameter_values["Cached"] = line.split(" ")[-1]
                elif line.find("Remove unpopular items") != -1:
                    parameter_values["RemoveUnpopularity"] = line.split(" ")[-1]
                elif line.find("Unpopular threshold") != -1:
                    parameter_values["UnpopularThreshold"] = line.split(" ")[-1]
                elif line.find("ROC_AUC") != -1:
                    results = json.loads(line[3:-1].replace("\'", "\""))

        if parameter_values["SolverType"].endswith("QPU") and parameter_values["Cached"] == "No":
            samples_filepath = os.path.join("/".join(filepath.split("/")[:-1]), "solver_responses.csv")
            samples_df = pd.read_csv(samples_filepath)
            parameter_values["CHAIN_BREAK_FRACTION_MAX"] = samples_df["chain_break_fraction"].max()
            parameter_values["CHAIN_BREAK_FRACTION_MEAN"] = np.mean(samples_df["chain_break_fraction"])
            parameter_values["CHAIN_BREAK_FRACTION_STD"] = np.std(samples_df["chain_break_fraction"])

        if parameter_values["SolverType"] == "LAZY_QPU":
            parameter_values["SolverType"] = "FIXED_QPU"

        if parameter_values["SolverType"].endswith("QPU") and parameter_values["SolverName"] == "":
            parameter_values["SolverName"] = "DW_2000Q"

        if parameter_values["FilterStrategy"] == "NONE":
            parameter_values["TopFilterValue"] = ""

        if parameter_values["SolverType"] == "SA":
            parameter_values["SolverName"] = ""
            parameter_values["ChainMultiplier"] = ""

        if parameter_values["RemoveUnpopularity"] == "False":
            parameter_values["UnpopularThreshold"] = 0

        df = df.append({
            "Datetime": filepath.split("/")[-2],
            "Algorithm": "Quantum SLIM",
            "SolverType": parameter_values["SolverType"],
            "SolverName": parameter_values["SolverName"],
            "Loss": parameter_values["Loss"],
            "AggregationStrategy": parameter_values["AggregationStrategy"],
            "FilterStrategy": parameter_values["FilterStrategy"],
            "TopFilterValue": parameter_values["TopFilterValue"],
            "TopK": parameter_values["TopK"],
            "NumReads": parameter_values["NumReads"],
            "AlphaMultiplier": parameter_values["AlphaMultiplier"],
            "ConstraintMultiplier": parameter_values["ConstraintMultiplier"],
            "ChainMultiplier": parameter_values["ChainMultiplier"],
            "UnpopularThreshold": parameter_values["UnpopularThreshold"],
            "Cached": parameter_values["Cached"],
            "PRECISION": results["PRECISION"],
            "RECALL": results["RECALL"],
            "MAP": results["MAP"],
            "NDCG": results["NDCG"],
            "AVERAGE_POPULARITY": results["AVERAGE_POPULARITY"],
            "DIVERSITY_MEAN_INTER_LIST": results["DIVERSITY_MEAN_INTER_LIST"],
            "COVERAGE_ITEM": results["COVERAGE_ITEM"],
            "CHAIN_BREAK_FRACTION_MAX": parameter_values["CHAIN_BREAK_FRACTION_MAX"],
            "CHAIN_BREAK_FRACTION_MEAN": parameter_values["CHAIN_BREAK_FRACTION_MEAN"],
            "CHAIN_BREAK_FRACTION_STD": parameter_values["CHAIN_BREAK_FRACTION_STD"],
        }, ignore_index=True)
    print(df.head())

    date_string = datetime.now().strftime('%b%d_%H-%M-%S')
    df.to_csv(os.path.join(FINAL_REPORT_PATH, "{}.csv".format(date_string)), index=False)
