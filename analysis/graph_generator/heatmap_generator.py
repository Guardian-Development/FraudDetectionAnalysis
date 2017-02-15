import seaborn as sns
import matplotlib.pyplot as plt


def generate_heat_map(matrix_data_set, output_location, filename, annotate=False, fig_size=(20, 20)):
    """
    Generates a heat map for the given matrix structured data_set.
    """
    plt.figure(figsize=fig_size)
    ax = plt.axes()
    gr = sns.heatmap(matrix_data_set, annot=annotate, fmt='g', ax=ax).get_figure()
    ax.set_title('Predicted class (X) against Actual class (Y)')
    gr.savefig(output_location + filename)


def generate_cluster_map(matrix_data_set, output_location, filename, annotate=False, fig_size=(20, 20)):
    """
    Generates a cluster map for the given matrix structured data_set.
    """
    gr = sns.clustermap(matrix_data_set, annot=annotate, figsize=fig_size)
    gr.savefig(output_location + filename)
