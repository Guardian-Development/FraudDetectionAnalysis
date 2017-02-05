import seaborn as sns
import matplotlib.pyplot as plt


def generate_heat_map(matrix_data_set, output_location, filename, annotate=False, fig_size=(20, 20)):
    """
    Generates a heat map for the given matrix structured data_set.
    """
    plt.figure(figsize=fig_size)
    gr = sns.heatmap(matrix_data_set, annot=annotate, fmt='g').get_figure()
    gr.savefig(output_location + filename)


def generate_cluster_map(matrix_data_set, output_location, filename, annotate=False, fig_size=(20, 20)):
    """
    Generates a cluster map for the given matrix structured data_set.
    """
    gr = sns.clustermap(matrix_data_set, annot=annotate, figsize=fig_size)
    gr.savefig(output_location + filename)
