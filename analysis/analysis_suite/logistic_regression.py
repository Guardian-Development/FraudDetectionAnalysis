from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def perform_logistic_regression(x_train, x_test, y_train, y_test):
    """
    Looks for the best values for logistic regression function,
    trains the model, then returns the model to you for analysis.
    :return: confusion matrix for model
    """
    log_model = LogisticRegression()
    log_model.fit(x_train, y_train)
    predictions = log_model.predict(x_test)

    print_classification_results_console(y_test, predictions)
    return confusion_matrix(y_test, predictions)


def print_classification_results_console(y_test, predictions):
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
