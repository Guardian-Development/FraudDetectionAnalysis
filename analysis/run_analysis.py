import os
from analysis.analysis_suite.analyser import Analyser, get_train_test_split
from analysis.analysis_suite.logistic_regression import perform_logistic_regression
from analysis.data_manager.file_reader import read_csv_file
from analysis.graph_generator.graph_builder import GraphBuilder


def run_initial_data_analysis():
    """
    Runs an initial look at the transaction data provided.
    """
    data_set = read_csv_file('/creditcard.csv', os.path.dirname(__file__))
    analyser = Analyser(data_set=data_set, prediction_column='Class') \
        .scale_column_to_range('Amount')

    # build data for all analysers
    x_train, x_test, y_train, y_test = get_train_test_split(analyser)

    confusion_matrix = perform_logistic_regression(x_train, x_test, y_train, y_test)

    GraphBuilder() \
        .with_output_location('./results/') \
        .with_heat_map(analyser.get_column_correlations(), 'field_correlations_heat_map.png') \
        .with_cluster_map(analyser.get_column_correlations(), 'field_correlations_cluster_map.png') \
        .with_heat_map(confusion_matrix, 'logistic_regression_attempt_1_heat_map.png', annotate=True, fig_size=(5, 5))


# entry point
if __name__ == '__main__':
    run_initial_data_analysis()
