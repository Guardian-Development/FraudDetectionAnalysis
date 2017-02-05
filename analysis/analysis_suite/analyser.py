from sklearn.preprocessing import StandardScaler


class Analyser(object):
    def scale_column_to_range(self, column_name):
        """
        Scales a given column to within a default range to allow for easier analysis.
        :param column_name: the column to scale
        :return: self
        """
        self.data_set[column_name] = \
            StandardScaler().fit_transform(self.data_set[column_name].values.reshape(-1, 1))
        return self

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

    def __init__(self, data_set, prediction_column):
        """
        Takes in the data_set you wish to perform analysis on
        :param data_set: data_set to analyse
        :param prediction_column: the name of the column you are trying to predict
        """
        self.data_set = data_set
        self.prediction_column = prediction_column
