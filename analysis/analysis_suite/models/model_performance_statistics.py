from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def print_statistics_on_grid_search(grid_search):
    """
    Prints statistics on the result of Grid Search finding the best parameters for a model
    :param grid_search: the fitted GridSearch object
    :return: null
    """
    print(grid_search.best_params_)
    print(grid_search.best_score_)


def get_confusion_matrix(actual, predictions):
    """
    Gets the confusion matrix for predicted results against actual results
    :param actual: actual results
    :param predictions: predicted results
    :return: confusion matrix
    """
    __print_statistics_on_predictions(actual, predictions)
    return confusion_matrix(actual, predictions)


def __print_statistics_on_predictions(actual, predictions):
    """
    Prints statistics about the result of the model against the actual classification
    :param actual: the actual classifications
    :param predictions: the predicted classifications
    """
    print(classification_report(actual, predictions))
    print(confusion_matrix(actual, predictions))
