#!/usr/local/bin/python

import numpy as np
from time import time
import sklearn.svm as svm
from sklearn.grid_search import GridSearchCV

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def func_grid_search():
    """
    This function does grid search to find out the best parameters.
    It returns a classifier object.
    """
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],}

    return GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'), param_grid)


def func_fitting(clf, x_training, y_training):
    """This function takes training data x_train and y_train, and produces a fitted classifier."""
    return clf.fit(x_training, y_training)


def func_SVM(x_train, y_train, x_test):
    # Train a SVM classification model

    print("Fitting the classifier to the training set")
    t0 = time()

    clf_svm = func_grid_search()
    clf_svm = func_fitting(clf_svm, x_train, y_train)

    print "done in %0.3fs" % (time() - t0)
    print "Best estimator found by grid search:"
    print clf_svm.best_estimator_
    # print "Best Score:", clf_svm.grid_scores_, '\n'

    # Quantitative evaluation of the model quality on the test set

    print("Making prodictions")
    t0 = time()
    y_pred = clf_svm.predict(x_test)
    print "done in %0.3fs" % (time() - t0)

    # print classification_report(y_test, y_pred, target_names=target_names)
    # print confusion_matrix(y_test, y_pred, labels=range(n_classes))

    return y_pred


def main():

    x_training = None
    y_training = None
    x_testing = None



    y_pred = func_SVM(x_training, y_training, x_testing)
    print y_pred


if __name__ == '__main__':

    x_training = np.array([[-1, -1, -1, -1], [-2, -1, -2, -3], [1, 1, 2, 1], [2, 1, 3, 3],
                           [1, 3, 1, 1], [2, 5, 3, 1], [-1, -3, -2, -1], [-2, -5, -3, -2]])
    y_training = np.array([1, 1, 2, 2, 1, 1, 2, 2])
    x_testing = [-0.8, -1, 2, -3]

    # print func_SVM(x_training, y_training, x_testing)

    y_pred = func_SVM(x_training, y_training, x_testing)
    print "The predicted class for testing point [-0.8, -1, 2, -3] is:", y_pred[0]
