import argparse
import os
from datetime import datetime


def cast_dict_elements(temp_dict):
    def represents_int(s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    def represents_float(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    for key, elem in temp_dict.items():
        if represents_int(elem):
            temp_dict[key] = int(elem)
        elif represents_float(elem):
            temp_dict[key] = float(elem)
    return temp_dict


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def handle_folder_creation(result_path: str, filename="results.txt", retrieve_text_file=True):
    """
    Handle the creation of a folder and return a file in which it is possible to write in that folder, moreover
    it also returns the path to the result path with current datetime appended

    :param result_path: basic folder where to store results
    :param filename: name of the result file to create
    :param retrieve_text_file: whether to retrieve a text file or not
    :return (descriptor to a file opened in write mode within result_path folder. The name of the file
    is result.txt, folder path with the date of the experiment). The file descriptor will be None
    retrieve_text_file is set to False
    """
    date_string = datetime.now().strftime('%b%d_%H-%M-%S/')
    output_folder_path = os.path.join(result_path, date_string)
    output_file_name = output_folder_path + filename
    try:
        if not os.path.exists(output_folder_path):
            os.mkdir(output_folder_path)
    except FileNotFoundError as e:
        os.makedirs(output_folder_path)

    fd = open(output_file_name, "w") if retrieve_text_file else None

    return fd, output_folder_path


def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return fig_width_in, fig_height_in


def save_low_and_high_resolution_images(filename):
    import matplotlib.pyplot as plt
    plt.savefig(filename + ".pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(filename + ".png", format="png", bbox_inches="tight", dpi=75)


def get_project_root_path():
    import os
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
