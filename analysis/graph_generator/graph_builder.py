from analysis.graph_generator.generic_generator import generate_pair_plot
from analysis.graph_generator.heatmap_generator import generate_heat_map, generate_cluster_map
import matplotlib.pyplot as plt


class GraphBuilder(object):
    def with_cluster_map(self, data_set, filename):
        """
        Generates a cluster map for the given data_set.
        :param data_set: the data_Set to use in the generation
        :param filename: the filename to save the graph with
        :return: self
        """
        generate_cluster_map(data_set, self.output_location, filename)
        return self

    def with_heat_map(self, data_set, filename):
        """
        Generates a heat map of the given data_set.
        :param data_set: the data_set to use in the generation
        :param filename: the filename to save the graph with
        :return: self
        """
        generate_heat_map(data_set, self.output_location, filename)
        return self

    def with_pair_plot(self, data_set, filename):
        """
        Generates a generic pair plot of the data set.
        :param data_set: the data_set you wish to plot
        :param filename: the filename to save the graph with
        :return: self
        """
        generate_pair_plot(data_set, self.output_location, filename)
        return self

    def with_graph_size(self, height, width):
        """
        Sets the generic graph size for all generated graphs.
        :param height: height of graph in inches
        :param width: width of graph in inches
        :return: self
        """
        plt.figure(figsize=(height, width))
        return self

    def with_output_location(self, output_location):
        """
        Sets the graph output location.
        :param output_location: the output location folder
        :return: self
        """
        self.output_location = output_location
        return self

    def __init__(self):
        """
        Initialises a builder than can be used to create multiple graphs.
        """
