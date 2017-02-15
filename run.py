import sys
import argparse

from analysis.analysis_runner import run_data_analysis


def start_analysis(file_location, predicting_col, scoring_param, columns_to_scale=None):
    """
    Starts the analysis suite for the given parameters.
    :param columns_to_scale: columns that will require scaling inline with rest of data set
    :param file_location: csv file location including name of file
    :param predicting_col: column containing the category we are trying to predict
    :param scoring_param: the scoring system to judge model efficiency
    :return: null
    """
    run_data_analysis(
        file_location=file_location,
        predicting_column=predicting_col,
        scoring_parameter=scoring_param,
        columns_to_scale=columns_to_scale)


def parse_command_line_arguments(argv):
    """
    Takes in array of command line arguments and parses them,
    :param argv: array
    :return: file_location, predicting_col, scoring_param, columns_to_scale
    """
    print("reading command line arguments in...")

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-i', '--input', help='Location of input csv file', required=True)
    parser.add_argument('-p', '--predicting', help='The column name containing the category to predict', required=True)
    parser.add_argument('-s', '--scoring', help='The scoring type to be used with model evaluation', required=False)
    parser.add_argument('-c', '--scale', help='List of column names to scale values for', nargs='+')
    args = parser.parse_args()

    return args.input, args.predicting, args.scoring, args.scale


def main():
    """
    Start application
    :return: none
    """
    file_location, predicting_col, scoring_param, columns_to_scale = parse_command_line_arguments(sys.argv[1:])
    start_analysis(file_location, predicting_col, scoring_param, columns_to_scale)


# entry point
if __name__ == '__main__':
    main()
