import os
from analysis.analysis_suite.analyser import Analyser, get_even_weighted_train_test_split, get_train_test_split
from analysis.analysis_suite.logistic_regression import LogisticRegressionModel, OptimisedLogisticRegressionModel
from analysis.data_manager.file_reader import read_csv_file
from analysis.graph_generator.graph_builder import GraphBuilder


def run_initial_data_analysis():
    """
    Runs analysis on data set.
    """
    data_set = read_csv_file('/creditcard.csv', os.path.dirname(__file__))
    analyser = Analyser(data_set=data_set, prediction_column='Class') \
        .scale_column_to_range('Amount')

    x_train_full_data, x_test_full_data, y_train_full_data, y_test_full_data = \
        get_train_test_split(analyser)

    x_train_even_split, x_test_even_split, y_train_even_split, y_test_even_split = \
        get_even_weighted_train_test_split(analyser)

    param_grid = {'C': [0.1, 1, 10, 100, 1000], 'penalty': ['l1', 'l2']}
    log_model = OptimisedLogisticRegressionModel()\
        .with_training_data(x_train_even_split, y_train_even_split)\
        .with_parameter_grid(param_grid)\
        .with_success_goal('f1')\
        .find_best_params()

    confusion_matrix = log_model.predict_logistic_regression(x_test_full_data, y_test_full_data)

    GraphBuilder() \
        .with_output_location('./results/') \
        .with_heat_map(analyser.get_column_correlations(), 'field_correlations_heat_map.png') \
        .with_cluster_map(analyser.get_column_correlations(), 'field_correlations_cluster_map.png') \
        .with_heat_map(confusion_matrix, 'logistic_regression_attempt_4_heat_map.png', annotate=True, fig_size=(5, 5))


# entry point
if __name__ == '__main__':
    run_initial_data_analysis()
