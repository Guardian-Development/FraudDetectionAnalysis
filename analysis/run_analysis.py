import os

from analysis.analysis_suite.data_manipulation.data_cleaner import scale_column_to_range
from analysis.analysis_suite.data_manipulation.file_reader import read_csv_file
from analysis.analysis_suite.data_manipulation.train_test_split_builder import TrainTestSplitBuilder
from analysis.analysis_suite.predictive_models_accessor import PredictiveModelsAccessor


def run_initial_data_analysis():
    """
    Runs analysis on data set.
    """
    data_frame = read_csv_file('/creditcard.csv', os.path.dirname(__file__))
    data_frame = scale_column_to_range(data_frame=data_frame, column_name='Amount')

    x_train_full_data, x_test_full_data, y_train_full_data, y_test_full_data = \
        TrainTestSplitBuilder()\
        .with_data_frame(data_frame=data_frame, predicting_column='Class')\
        .split_data()

    x_train_even_split, x_test_even_split, y_train_even_split, y_test_even_split = \
        TrainTestSplitBuilder()\
        .with_data_frame(data_frame=data_frame, predicting_column='Class')\
        .use_column_distribution_split()\
        .split_data()

    confusion_matrix = \
        PredictiveModelsAccessor()\
        .with_scoring_goal(scoring_parameter='f1')\
        .optimise_and_train_models(x_train_even_split, y_train_even_split)\
        .test_predictions(x_test_full_data, y_test_full_data)

    print(confusion_matrix)

# entry point
if __name__ == '__main__':
    run_initial_data_analysis()
