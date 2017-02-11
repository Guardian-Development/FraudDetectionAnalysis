from sklearn.model_selection import GridSearchCV
from analysis.analysis_suite.models.logistic_regression import LogisticRegressionModelEnhancer
from analysis.analysis_suite.models.model_performance_statistics import print_statistics_on_grid_search, \
    get_confusion_matrix
from collections import Counter


class PredictiveModelsAccessor(object):

    def optimise_and_train_models(self, x_train, y_train):
        """
        For each of the model enhancers available, find the best parameters available for it
        to use with predictions then save the model itself in fitted_models
        :param x_train: x training data array
        :param y_train: y training data array
        :return: self
        """
        for model_enhancer in self.available_model_enhancers:
            grid = GridSearchCV(model_enhancer.get_model(),
                                model_enhancer.get_optimisation_parameters(),
                                scoring=self.scoring_parameter)

            grid.fit(x_train, y_train)
            print_statistics_on_grid_search(grid)
            self.fitted_models.append(grid.best_estimator_)

        return self

    def test_predictions(self, x_test, y_test):
        """
        Tests the models ability to predict classification based on x_test and y_test
        :param x_test: data frame
        :param y_test: data frame
        :return: confusion matrix
        """
        predictions = []

        for data_row in x_test.iterrows():
            prediction = self.__get_prediction__(data_row)
            predictions.append(prediction)

        return get_confusion_matrix(y_test, predictions)

    def with_scoring_goal(self, scoring_parameter):
        """
        Sets the scoring parameter to be used when judging the success of the model
        :param scoring_parameter: string
        :return: self
        """
        self.scoring_parameter = scoring_parameter
        return self

    def __get_prediction__(self, data_row):
        """
        Gets the prediction of the classification for the data row
        :param data_row: data frame
        :return: class prediction
        """
        model_predictions = []

        # element 0 holds the row number which we do not require.
        data_row = data_row[1].values.reshape(1, -1)
        for model in self.fitted_models:
            prediction = model.predict(data_row)
            model_predictions.append(prediction[0])

        if len(set(model_predictions)) == 1:
            return model_predictions[0]
        else:
            prediction_counts = Counter(model_predictions)
            return prediction_counts.most_common(1)[0]

    def __init__(self):
        self.fitted_models = []
        self.available_model_enhancers = [LogisticRegressionModelEnhancer()]
