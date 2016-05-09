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


def func_SVM(x_train, y_train, x_test, verbose=False):


    ### Training a SVM classification model
    if verbose:
        print("Fitting the classifier to the training set")
        t0 = time()

    clf_svm = func_grid_search()
    clf_svm = func_fitting(clf_svm, x_train, y_train)

    if verbose:
        print "done in %0.3fs" % (time() - t0)
        print "Best estimator found by grid search:"
        print clf_svm.best_estimator_
        # print "Best Score:", clf_svm.grid_scores_, '\n'
        print("Making prodictions")
        t0 = time()

    y_pred = clf_svm.predict(x_test)

    if verbose:
        print "done in %0.3fs" % (time() - t0)

    return y_pred


def func_prediction_for_each_set(train, test):

    ### Reshaping training data
    x_training = [train[i, j] for i in range(train.shape[0]) for j in range(train.shape[1])]
    y_training = [i for i in range(1, train.shape[0] + 1) for j in range(train.shape[1])]

    ### Reshaping testing data
    x_testing = [test[i, j] for i in range(test.shape[0]) for j in range(test.shape[1])]
    y_testing = np.array([i for i in range(1, test.shape[0] + 1) for j in range(test.shape[1])])

    ### Running SVM and make predictions
    y_pred = func_SVM(x_training, y_training, x_testing)

    return y_pred, y_testing


def svm_for_egf_unnorl():
    egf_comp = [6, 10, 18, 50, 100]

    for egf_comp in egf_comp:

        pred_mat = None
        test_mat = None

        for i in range(13):

            ### Loading feature extracted data
            train = np.load('../../Data/eigenfaces/unnormalized/%d_component/egf_%d_tr.npy' % (egf_comp, i))
            test = np.load('../../Data/eigenfaces/unnormalized/%d_component/egf_%d_te.npy' % (egf_comp, i))

            ### Making predictions
            y_pred, y_testing = func_prediction_for_each_set(train, test)

            ### Collecting data
            if pred_mat is None:
                pred_mat = y_pred
                test_mat = y_testing
            else:
                pred_mat = np.vstack((pred_mat, y_pred))
                test_mat = np.vstack((test_mat, y_testing))

        print pred_mat.shape
        print test_mat.shape

        print pred_mat[0]
        print test_mat[0]

        np.save('predictions/egf_unnorl_%d_comp_pred' % egf_comp, pred_mat)
        np.save('predictions/egf_unnorl_%d_comp_true_label' % egf_comp, test_mat)


def svm_for_egf_norl():
    egf_comp = [8, 14, 24, 50, 100]

    for egf_comp in egf_comp:

        pred_mat = None
        test_mat = None

        for i in range(13):

            ### Loading feature extracted data
            train = np.load('../../Data/eigenfaces/normalized/%d_component/egf_%d_tr.npy' % (egf_comp, i))
            test = np.load('../../Data/eigenfaces/normalized/%d_component/egf_%d_te.npy' % (egf_comp, i))

            ### Making predictions
            y_pred, y_testing = func_prediction_for_each_set(train, test)

            ### Collecting data
            if pred_mat is None:
                pred_mat = y_pred
                test_mat = y_testing
            else:
                pred_mat = np.vstack((pred_mat, y_pred))
                test_mat = np.vstack((test_mat, y_testing))

        print pred_mat.shape
        print test_mat.shape

        print pred_mat[0]
        print test_mat[0]

        np.save('predictions/egf_norl_%d_comp_pred' % egf_comp, pred_mat)
        np.save('predictions/egf_norl_%d_comp_true_label' % egf_comp, test_mat)


def svm_for_egf_norl_r3():
    egf_comp = [8, 14, 24, 50, 100]

    for egf_comp in egf_comp:

        pred_mat = None
        test_mat = None

        for i in range(13):

            ### Loading feature extracted data
            train = np.load('../../Data/eigenfaces/normalized_first_3_components_removed/%d_component/egf_%d_tr.npy' % (egf_comp, i))
            test = np.load('../../Data/eigenfaces/normalized_first_3_components_removed/%d_component/egf_%d_te.npy' % (egf_comp, i))

            ### Making predictions
            y_pred, y_testing = func_prediction_for_each_set(train, test)

            ### Collecting data
            if pred_mat is None:
                pred_mat = y_pred
                test_mat = y_testing
            else:
                pred_mat = np.vstack((pred_mat, y_pred))
                test_mat = np.vstack((test_mat, y_testing))

        print pred_mat.shape
        print test_mat.shape

        print pred_mat[0]
        print test_mat[0]

        np.save('predictions/egf_norl_r3_%d_comp_pred' % egf_comp, pred_mat)
        np.save('predictions/egf_norl_r3_%d_comp_true_label' % egf_comp, test_mat)


def svm_for_nmf():
    egf_comp = [8, 10, 14, 24, 50]

    for egf_comp in egf_comp:

        pred_mat = None
        test_mat = None

        for i in range(13):

            ### Loading feature extracted data
            train = np.load('../../Data/nmf/%d/nmf_%d_tr.npy' % (egf_comp, i))
            test = np.load('../../Data/nmf/%d/nmf_%d_te.npy' % (egf_comp, i))

            print train.shape
            print test.shape

            # print train

            ### Making predictions
            y_pred, y_testing = func_prediction_for_each_set(train, test)

            ### Collecting data
            if pred_mat is None:
                pred_mat = y_pred
                test_mat = y_testing
            else:
                pred_mat = np.vstack((pred_mat, y_pred))
                test_mat = np.vstack((test_mat, y_testing))

        print pred_mat.shape
        print test_mat.shape

        print pred_mat[0]
        print test_mat[0]

        np.save('predictions/nmf_%d_comp_pred' % egf_comp, pred_mat)
        np.save('predictions/nmf_%d_comp_true_label' % egf_comp, test_mat)


def main():

    egf_comp = [6, 10, 20, 50, 100]

    for egf_comp in egf_comp:

        pred_mat = None
        test_mat = None

        for i in range(12):

            ### Loading feature extracted data
            train = np.load('../../Data/eigenfaces/%d_component/egf_%d_tr.npy' % (egf_comp, i))
            test = np.load('../../Data/eigenfaces/%d_component/egf_%d_te.npy' % (egf_comp, i))

            ### Making predictions
            y_pred, y_testing = func_prediction_for_each_set(train, test)

            ### Collecting data
            if pred_mat is None:
                pred_mat = y_pred
                test_mat = y_testing
            else:
                pred_mat = np.vstack((pred_mat, y_pred))
                test_mat = np.vstack((test_mat, y_testing))

        print pred_mat.shape
        print test_mat.shape

        print pred_mat[0]
        print test_mat[0]

        np.save('predictions/egf_%d_comp_pred' % egf_comp, pred_mat)
        np.save('predictions/egf_%d_comp_true_label' % egf_comp, test_mat)

if __name__ == '__main__':

    # main()

    # svm_for_egf_unnorl()
    # svm_for_egf_norl()
    # svm_for_egf_norl_r3()

    svm_for_nmf()

    ### Testing Code
    # x_training = np.array([[-1, -1, -1, -1], [-2, -1, -2, -3], [1, 1, 2, 1], [2, 1, 3, 3],
    #                        [1, 3, 1, 1], [2, 5, 3, 1], [-1, -3, -2, -1], [-2, -5, -3, -2]])
    # y_training = np.array([1, 1, 2, 2, 1, 1, 2, 2])
    # x_testing = [-0.8, -1, 2, -3]
    #
    # # print func_SVM(x_training, y_training, x_testing)
    #
    # y_pred = func_SVM(x_training, y_training, x_testing)
    # print "The predicted class for testing point [-0.8, -1, 2, -3] is:", y_pred[0]
