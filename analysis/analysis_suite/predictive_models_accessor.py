# PredictiveModelsAccessor:
# 	array of available models, initialised on init
#
# 	OptimiseAndTrainModels(x_train, y_train): for each model in array, gets the available params to test for it,
#     wraps it in grid search, then saves back the optimised model to the 		dictionary (so we can store accuracy etc with it)
#
# 	Predict(x_test, y_test): predicts for each model then if there is a discrepency in 	classification take most accurate model (for now) r
#     return confusion matrix -> through 		ModelPerformanceStatistics
from sklearn.model_selection import GridSearchCV

from analysis.analysis_suite.models.logistic_regression import LogisticRegressionModelEnhancer
from analysis.analysis_suite.models.model_performance_statistics import print_statistics_on_grid_search


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
                                model_enhancer.optimisation_parameters,
                                scoring=self.scoring_parameter)

            grid.fit(x_train, y_train)
            print_statistics_on_grid_search(grid)
            self.fitted_models.append(grid.best_estimator_)

        return self

    def test_predictions(self, x_test, y_test):
        predictions = []

        for data_row in x_test:
            model_results = []

            for model in self.fitted_models:
                model_results.append(model.predict(data_row))

            # TODO: get unique results from model_results
            # print what each model predicted
            # set predictions to be most common one
            # return confusion matrix for predictions against y_test

    def __init__(self):
        # TODO: pass scoring parameter
        self.fitted_models = []
        self.available_model_enhancers = [LogisticRegressionModelEnhancer()]
        self.scoring_parameter = 'f1'
