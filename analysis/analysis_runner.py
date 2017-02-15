import datetime
import time

from analysis.analysis_suite.data_manipulation.data_cleaner import scale_column_to_range
from analysis.analysis_suite.data_manipulation.file_reader import read_csv_file
from analysis.analysis_suite.data_manipulation.train_test_split_builder import TrainTestSplitBuilder
from analysis.analysis_suite.predictive_models_accessor import PredictiveModelsAccessor
from analysis.graph_generator.heatmap_generator import generate_heat_map


# TODO: read in csv file location and predicting column from command line
# TODO: optional output parameter of where you want the resulting graph
# TODO: optional scoring parameter to be passed
# TODO: refactor code around train test split to see if can make cleaner

def run_data_analysis(file_location, predicting_column, scoring_parameter):
    """
    Runs analysis on data set.
    """
    print("reading data into program...")
    current_date_time_string = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    data_frame = read_csv_file(file_location)
    data_frame = scale_column_to_range(data_frame=data_frame, column_name='Amount')

    print("building train, test data sets...")
    x_train, x_test, y_train, y_test = \
        TrainTestSplitBuilder() \
            .with_data_frame(data_frame=data_frame, predicting_column=predicting_column) \
            .use_column_distribution_split_boundary() \
            .split_data()

    print("starting timer...")
    start = time.time()

    print("setting up predictive models accessor...")
    confusion_matrix = \
        PredictiveModelsAccessor() \
            .with_scoring_goal(scoring_parameter=scoring_parameter) \
            .optimise_and_train_models(x_train, y_train) \
            .test_predictions(x_test, y_test)

    print("stopping timer...")
    end = time.time()

    generate_heat_map(
        matrix_data_set=confusion_matrix,
        output_location='./results/',
        filename=current_date_time_string + '_results.png',
        annotate=True,
        fig_size=(5, 5))

    generate_heat_map(
        matrix_data_set=confusion_matrix,
        output_location='./results/',
        filename='latest_result.png',
        annotate=True,
        fig_size=(7, 7))

    print("time taken to predict all data: ", (end - start))

