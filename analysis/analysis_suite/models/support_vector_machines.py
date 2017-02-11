from sklearn.svm import SVC


class SupportVectorMachinesModelEnhancer(object):
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

    def get_accuracy(self):
        """
        Gets the known accuracy of the model which should be set after GridSearch is ran
        :return: decimal between 0 and 1
        """
        return self.accuracy

    def set_accuracy(self, accuracy):
        """
        Sets the accuracy of the model which is calculated through GridSearc
        :param accuracy: decimal between 0 and 1
        :return: self
        """
        self.accuracy = accuracy
        return self

    def __init__(self):
        self.model = SVC()
        self.accuracy = 0
        self.optimisation_parameters = {
            'C': [0.1, 1, 5],
            'random_state': [101]}

