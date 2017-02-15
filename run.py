import sys
import getopt

from analysis.analysis_runner import run_data_analysis


def parse_command_line_arguments(argv):
    file_location = None
    predicting_col = None
    scoring_param = None

    try:
        opts, args = getopt.getopt(argv, "hi:p:s")
    except getopt.GetoptError:
        print("run.py -i <input_file> -p <predicting_column> -s <scoring_parameter>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print("run.py -i <input_file> -p <predicting_column> -s <scoring_parameter>")
            sys.exit()
        elif opt == '-i':
            file_location = arg
        elif opt == '-p':
            predicting_col = arg
        elif opt == '-s':
            scoring_param = arg

    if None in [file_location, predicting_col, scoring_param]:
        raise ValueError("must specify all command line arguments. use -h for help")

    return file_location, predicting_col, scoring_param


def start_analysis(file_location, predicting_col, scoring_param):
    run_data_analysis(
        file_location=file_location,
        predicting_column=predicting_col,
        scoring_parameter=scoring_param)


def main():
    file_location, predicting_col, scoring_param = parse_command_line_arguments(sys.argv[1:])
    start_analysis(file_location, predicting_col, scoring_param)


# entry point
if __name__ == '__main__':
    # location = os.path.dirname(__file__) + '/creditcard.csv'
    # predicting_col = 'Class'
    # scoring_param = 'f1'
    main()
