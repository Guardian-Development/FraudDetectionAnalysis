import os
from analysis.analysis_suite.analyser import Analyser
from analysis.data_manager.file_reader import read_csv_file
from analysis.graph_generator.graph_builder import GraphBuilder


def run_initial_data_analysis():
    """
    Runs an initial look at the transaction data provided.
    """
    data_set = read_csv_file('/creditcard.csv', os.path.dirname(__file__))
    analyser = Analyser(data_set=data_set, prediction_column='Class')\
        .scale_column_to_range('Amount')

    GraphBuilder()\
        .with_output_location('./results/')\
        .with_graph_size(15, 12)\
        .with_heat_map(analyser.get_column_correlations(), 'field_correlations_heat_map.png')\
        .with_cluster_map(analyser.get_column_correlations(), 'field_correlations_cluster_map.png')


# entry point
if __name__ == '__main__':
    run_initial_data_analysis()
