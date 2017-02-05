import seaborn as sns
import matplotlib.pyplot as plt


def generate_heat_map(matrix_data_set, output_location, filename, annotate=False):
    """
    Generates a heat map for the given matrix structured data_set.
    """
    plt.figure(figsize=(20, 20))
    gr = sns.heatmap(matrix_data_set, annot=annotate).get_figure()
    gr.savefig(output_location + filename)


def generate_cluster_map(matrix_data_set, output_location, filename, annotate=False):
    """
    Generates a cluster map for the given matrix structured data_set.
    """
    gr = sns.clustermap(matrix_data_set, annot=annotate, figsize=(20, 20))
    gr.savefig(output_location + filename)
