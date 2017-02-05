from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def get_train_test_split(analyser):
    """
    Performs a train test split on a data structure held by an Analyser object.
    :param analyser: the analyser to use for the data source
    :return: x_train, x_test, y_train, y_test
    """
    y = analyser.data_set[analyser.prediction_column]
    x = analyser.data_set.drop(analyser.prediction_column, axis=1)
    return train_test_split(x, y, test_size=0.4, random_state=101)


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
