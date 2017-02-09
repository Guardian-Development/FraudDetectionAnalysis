from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


def print_classification_results_console(y_test, predictions):
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))


class LogisticRegressionModel(object):
    def set_logistic_regression_model(self, model):
        """
        optionally sets the logistic regression model from an
        external source
        :param model: the logistic regression model
        :return: self
        """
        self.log_model = model
        return self

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


class OptimisedLogisticRegressionModel(object):
    def find_best_params(self):
        """
        Finds the best parameters for the model then returns itself.
        :return: Logistic Regression Model
        """
        grid = GridSearchCV(self.log_model, self.parameter_grid, scoring=self.scoring_param)
        grid.fit(self.x_train, self.y_train)

        print(grid.best_params_)
        print(grid.best_score_)

        return LogisticRegressionModel() \
            .set_logistic_regression_model(grid.best_estimator_)

    def with_success_goal(self, scoring_param):
        """
        Sets the success criteria for the grid search when picking the
        best combination of parameters.
        :param scoring_param: the success criteria
        :return: self
        """
        self.scoring_param = scoring_param
        return self

    def with_training_data(self, x_train, y_train):
        """
        The training data to be used
        :param x_train:
        :param y_train:
        :return:
        """
        self.x_train = x_train
        self.y_train = y_train
        return self

    def with_parameter_grid(self, parameter_grid):
        """
        Takes a dictionary of parameters you want to test and their values.
        :param parameter_grid: paramter -> array value, pairs
        :return: self
        """
        self.parameter_grid = parameter_grid
        return self

    def __init__(self):
        """
        Creates the logistic regression model which will
        then be wrapped in a GridSearch object on run.
        """
        self.log_model = LogisticRegression()
