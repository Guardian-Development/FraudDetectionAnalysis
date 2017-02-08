from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


def get_train_test_split(analyser):
    """
    Performs a train test split on a data structure held by an Analyser object.
    :param analyser: the analyser to use for the data source
    :return: x_train, x_test, y_train, y_test
    """
    y = analyser.data_set[analyser.prediction_column]
    x = analyser.data_set.drop(analyser.prediction_column, axis=1)
    return train_test_split(x, y, test_size=0.4, random_state=101)


def get_even_weighted_train_test_split(analyser):
    """
    Performs a train test split on a data structure so you are left with
    an event split for the classes you are trying to predict.
    It randomly selects records to allow this to happen.
    :param analyser: the analyser to use for the data source
    :return: x_train, x_test, y_train, y_test
    """
    categories = analyser.data_set[analyser.prediction_column].unique()
    minimum_count = get_minimum_count_for_even_split(analyser, categories)
    chosen_indices_by_category = get_indices_by_category(analyser, categories, minimum_count)
    sampled_indices = np.concatenate(chosen_indices_by_category)
    under_sample_data = analyser.data_set.iloc[sampled_indices, :]
    print_resulting_data_frame_info(analyser, categories, under_sample_data)

    y = under_sample_data[analyser.prediction_column]
    x = under_sample_data.drop(analyser.prediction_column, axis=1)
    return train_test_split(x, y, test_size=0.4, random_state=101)


def get_minimum_count_for_even_split(analyser, categories):
    """
    Gets the minimum count by category so we know what the minimum category count is
    :param analyser: the data source
    :param categories: the categories that are available
    :return: int
    """
    minimum_count = 0
    for category in categories:
        rows_found = analyser.data_set[analyser.data_set[analyser.prediction_column] == category]
        count = len(rows_found)

        if count < minimum_count or minimum_count == 0:
            minimum_count = count
    return minimum_count


def print_resulting_data_frame_info(analyser, categories, under_sample_data):
    """
    Prints the total number of rows used to get an even split of data
    :param analyser: the data source
    :param categories:  the categories that are available
    :param under_sample_data: the data that will be used in the even split
    :return: null
    """
    for category in categories:
        print("Percentage of category in data: ",
              len(under_sample_data[under_sample_data[analyser.prediction_column] == category]) / len(
                  under_sample_data))
    print("Total number of transactions used: ", len(under_sample_data))


def get_indices_by_category(analyser, categories, minimum_count):
    """
    Gets an array of row indices to be included in the sample data to provide an even category split
    :param analyser: the data source
    :param categories: the categories that are available
    :param minimum_count: the amount of rows for each category we should include
    :return: array of int
    """
    chosen_indices_by_category = []
    for category in categories:
        category_indices = analyser.data_set[analyser.data_set[analyser.prediction_column] == category].index
        random_indices = np.random.choice(category_indices, minimum_count, replace=False)
        category_indices_array = np.array(random_indices)
        chosen_indices_by_category.append(category_indices_array)
    return chosen_indices_by_category


class Analyser(object):
    def get_column_correlations(self, include_prediction=False):
        """
        Computes the pairwise correlation of columns for the given data_set
        :return: DataSet of correlations
        """
        if include_prediction:
            return self.data_set.corr()

        data_set = self.data_set.drop(self.prediction_column, axis=1).corr()
        return data_set

    def get_current_data_set(self, include_prediction=True):
        """
        Returns the current state of the data_set
        """
        if include_prediction:
            return self.data_set

        data_set = self.data_set.drop(self.prediction_column, axis=1)
        return data_set

    def scale_column_to_range(self, column_name):
        """
        Scales a given column to within a default range to allow for easier analysis.
        :param column_name: the column to scale
        :return: self
        """
        self.data_set[column_name] = \
            StandardScaler().fit_transform(self.data_set[column_name].values.reshape(-1, 1))
        return self

    def __init__(self, data_set, prediction_column):
        """
        Takes in the data_set you wish to perform analysis on
        :param data_set: data_set to analyse
        :param prediction_column: the name of the column you are trying to predict
        """
        self.data_set = data_set
        self.prediction_column = prediction_column
