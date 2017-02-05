import matplotlib.pyplot as plt
import seaborn as sns


def generate_heat_map(matrix_data_set, output_location, filename, annotate=False):
    """
    Generates a heat map for the given matrix structured data_set.
    """
    gr = sns.heatmap(matrix_data_set, annot=annotate).get_figure()
    gr.savefig(output_location + filename)
    plt.clf()


def generate_cluster_map(matrix_data_set, output_location, filename, annotate=False):
    """
    Generates a cluster map for the given matrix structured data_set.
    """
    gr = sns.clustermap(matrix_data_set, annot=annotate)
    gr.savefig(output_location + filename)
    plt.clf()
