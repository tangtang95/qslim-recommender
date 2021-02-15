import argparse
import os

from scripts.experiments.run_quantum_slim import DEFAULT_OUTPUT_FOLDER, \
    run_experiment, save_result, parse_results_file
from src.utils.utilities import str2bool, cast_dict_elements


def get_arguments():
    """
    Defining the arguments available for the script

    :return: argument parser
    """
    parser = argparse.ArgumentParser()

    # Data setting
    parser.add_argument("-f", "--foldername", help="Folder name of the failed experiment in quantum_slim folder",
                        required=True, type=str)

    # Store results
    parser.add_argument("-v", "--verbose", help="Whether to output on command line much as possible",
                        type=str2bool, default=True)
    parser.add_argument("-sr", "--save_result", help="Whether to store results or not", type=str2bool, default=True)
    parser.add_argument("-o", "--output_folder", default=DEFAULT_OUTPUT_FOLDER,
                        help="Basic folder where to store the output", type=str)

    # Others
    parser.add_argument("-t", "--token", help="Token string in order to use DWave Sampler", type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    resumed_exp_foldername = args.foldername

    folderpath = os.path.join(DEFAULT_OUTPUT_FOLDER, args.foldername)

    exp_args_dict = parse_results_file(os.path.join(folderpath, "results_fail.txt"))
    exp_args_dict = cast_dict_elements(exp_args_dict)

    args_dict: dict = vars(args)

    final_dict = {**args_dict, **exp_args_dict}
    final_args = argparse.Namespace(**final_dict)

    mdl, result = run_experiment(final_args, do_preload=True, do_fit=True)
    print("Results: {}".format(str(result)))

    if args.save_result:
        result_folder = save_result(mdl, result, final_args)
        open(os.path.join(result_folder, "resumed_from_%s" % resumed_exp_foldername), 'a').close()





