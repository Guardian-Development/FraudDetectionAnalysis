from sklearn.model_selection import GridSearchCV
from analysis.analysis_suite.models.logistic_regression import LogisticRegressionModelEnhancer
from analysis.analysis_suite.models.model_performance_statistics import print_statistics_on_grid_search, \
    get_confusion_matrix
from collections import Counter
from analysis.analysis_suite.models.random_forest import RandomForestClassifierModelEnhancer
from analysis.analysis_suite.models.support_vector_machines import SupportVectorMachinesModelEnhancer


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
            model_enhancer.set_accuracy(grid.best_score_)
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
            return prediction_counts.most_common(1)[0][0]
            # return self.__resolve_prediction_conflict__(model_predictions)

    def __resolve_prediction_conflict__(self, model_predictions):
        """
        Given an array of predictions in the order of the models predicting them
        work out through their accuracy score the most accurate prediction
        :param model_predictions: array
        :return: prediction
        """
        prediction_score = {}
        for index, prediction in enumerate(model_predictions):
            score = prediction_score.get(prediction,
                                         {'accuracy_score': self.available_model_enhancers[index].get_accuracy(),
                                          'count': 1})
            if score.get('count', 0) > 1:
                score['accuracy_score'] = score.get('accuracy_score', 0) + self.available_model_enhancers[
                    index].get_accuracy()
                score['count'] = score.get('count', 0) + 1
                prediction_score[prediction] = score

        most_accurate_precision_prediction = 0
        most_accurate_precision = 0
        for prediction, stats in prediction_score.items():
            accuracy = stats.get('accuracy_score', 0)
            count = stats.get('count', 0)
            precision = accuracy / count
            if precision > most_accurate_precision:
                most_accurate_precision_prediction = prediction

        return most_accurate_precision_prediction

    def __init__(self):
        self.fitted_models = []
        self.available_model_enhancers = [
            LogisticRegressionModelEnhancer(),
            RandomForestClassifierModelEnhancer(),
            SupportVectorMachinesModelEnhancer()
        ]
