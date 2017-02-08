from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def print_classification_results_console(y_test, predictions):
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))


class LogisticRegressionModel(object):
    def train_logistic_regression(self, x_train, y_train):
        """
        Train the logistic regression model on the data provided
        :param x_train: training data
        :param y_train: training data
        :return: self
        """
        self.log_model.fit(x_train, y_train)
        return self

    def predict_logistic_regression(self, x_test, y_test):
        """
        Performs the prediction based on the logistic regression model associated with this class.
        You should perform train_logistic_regression first.
        :param x_test: test data
        :param y_test: test data
        :return: confusion matrix of results of model against test data.
        """
        self.predictions = self.log_model.predict(x_test)
        print_classification_results_console(y_test, self.predictions)
        return confusion_matrix(y_test, self.predictions)

    def __init__(self):
        """
        Creates the logistic regression model ready for use.
        """
        self.log_model = LogisticRegression()
