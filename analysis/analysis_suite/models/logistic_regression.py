from sklearn.linear_model import LogisticRegression


class LogisticRegressionModelEnhancer(object):

    def get_model(self):
        """
        Gets a Logistic Regression Model
        :return: LogisticRegression
        """
        return self.model

    def get_optimisation_parameters(self):
        """
        Gets the optimisation parameters to be used when optimising this model
        :return: dict
        """
        return self.optimisation_parameters

    def __init__(self):
        self.model = LogisticRegression()
        self.optimisation_parameters = {'C': [0.1, 1, 10, 100, 1000], 'penalty': ['l1', 'l2']}
