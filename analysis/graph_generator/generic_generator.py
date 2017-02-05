import matplotlib.pyplot as plt
import seaborn as sns


def generate_pair_plot(data_set, output_location, filename):
    """
    Generates a Pair plot for the given data_set.
    """
    gr = sns.pairplot(data_set)
    gr.savefig(output_location + filename)
    plt.clf()
