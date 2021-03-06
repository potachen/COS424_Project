#!/usr/local/bin/python

import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import zero_one_loss


def plot_conf_mat(data, num=1, cmap=plt.cm.GnBu):
    plt.style.use('ggplot')

    plt.figure(num=None, figsize=[10, 8])
    plt.subplot(111)
    plt.subplots_adjust(top=0.96, bottom=0.08, right=1.05, left=0.08)

    plt.pcolor(data, cmap=cmap, vmin=0, vmax=1)
    # plt.pcolor(data, cmap=cmap, vmin=-200, vmax=200)
    # plt.pcolor(cm)

    plt.colorbar()
    # plt.title('Confusion Matrix', fontsize=18)
    plt.ylabel('True Subject', fontsize=16)
    plt.xlabel('Predicted Subject', fontsize=16)

    # tick_marks = np.arange(len(iris.target_names))
    # plt.xticks(tick_marks, iris.target_names, rotation=45)
    # plt.yticks(tick_marks, iris.target_names)

    tick_marks = np.arange(1, 11)
    plt.xticks(tick_marks - 0.5, tick_marks)#, rotation=45)
    plt.yticks(tick_marks - 0.5, tick_marks)

    # plt.savefig('../FIGs/plot_heatmap.pdf', format='pdf')
    # plt.savefig('../FIGs/plot_heatmap.png', format='png')

    # plt.show()


def plot_conf_mat_diff(data1, data2, num=1, cmap=plt.cm.GnBu):

    print np.sum(data1, axis=1)
    print np.sum(data2, axis=1)

    plt.style.use('ggplot')

    plt.figure(num=None, figsize=[28, 8])
    plt.subplot(131)
    # plt.subplots_adjust(top=0.96, bottom=0.08, right=1.05, left=0.08)
    plt.subplots_adjust(top = 0.94, bottom = 0.11, right = 1, left = 0.04)

    plt.pcolor(data1, cmap=cmap, vmin=0, vmax=676)
    # plt.pcolor(data1, cmap=cmap, vmin=-200, vmax=200)
    # plt.pcolor(cm)

    plt.colorbar()
    plt.title('PCA', fontsize=36)
    # plt.title('Confusion Matrix', fontsize=18)
    plt.ylabel('True Subject', fontsize=30)
    plt.xlabel('Predicted Subject', fontsize=30)

    # tick_marks = np.arange(len(iris.target_names))
    # plt.xticks(tick_marks, iris.target_names, rotation=45)
    # plt.yticks(tick_marks, iris.target_names)

    tick_marks = np.arange(1, 11)
    plt.xticks(tick_marks - 0.5, tick_marks, fontsize=20)#, rotation=45)
    plt.yticks(tick_marks - 0.5, tick_marks, fontsize=20)




    ### 2nd Part
    plt.subplot(132)
    plt.pcolor(data2, cmap=cmap, vmin=0, vmax=676)
    # plt.pcolor(cm)

    plt.colorbar()
    plt.title('NMF', fontsize=36)
    # plt.title('Confusion Matrix', fontsize=18)
    plt.ylabel('|', fontsize=50)
    plt.xlabel('Predicted Subject', fontsize=30)

    # tick_marks = np.arange(len(iris.target_names))
    # plt.xticks(tick_marks, iris.target_names, rotation=45)
    # plt.yticks(tick_marks, iris.target_names)

    tick_marks = np.arange(1, 11)
    plt.xticks(tick_marks - 0.5, tick_marks, fontsize=20)  # , rotation=45)
    plt.yticks(tick_marks - 0.5, tick_marks, fontsize=20)



    ### 3rd Part
    plt.subplot(133)
    plt.pcolor(data1 - data2, cmap=plt.cm.bwr, vmin=-338, vmax=338)
    # plt.pcolor(cm)

    plt.colorbar()
    plt.title('Difference', fontsize=36)
    # plt.title('Confusion Matrix', fontsize=18)
    plt.ylabel('||', fontsize=50)
    plt.xlabel('Predicted Subject', fontsize=30)

    # tick_marks = np.arange(len(iris.target_names))
    # plt.xticks(tick_marks, iris.target_names, rotation=45)
    # plt.yticks(tick_marks, iris.target_names)
    tick_marks = np.arange(1, 11)
    plt.xticks(tick_marks - 0.5, tick_marks, fontsize=20)  # , rotation=45)
    plt.yticks(tick_marks - 0.5, tick_marks, fontsize=20)

    plt.tight_layout()

    return plt



def plot_conf_mat_triple(data1, data2, data3, num=1, cmap=plt.cm.GnBu):

    # print np.sum(data1, axis=1)
    # print np.sum(data2, axis=1)

    plt.style.use('ggplot')

    plt.figure(num=None, figsize=[28, 8])
    plt.subplot(131)
    # plt.subplots_adjust(top=0.96, bottom=0.08, right=1.05, left=0.08)
    plt.subplots_adjust(top = 0.94, bottom = 0.11, right = 1, left = 0.04)

    plt.pcolor(data1, cmap=cmap, vmin=0, vmax=676)
    # plt.pcolor(data1, cmap=cmap, vmin=-200, vmax=200)
    # plt.pcolor(cm)

    plt.colorbar()
    plt.title('14 Components', fontsize=36)
    # plt.title('Confusion Matrix', fontsize=18)
    plt.ylabel('True Subject', fontsize=30)
    plt.xlabel('Predicted Subject', fontsize=30)

    # tick_marks = np.arange(len(iris.target_names))
    # plt.xticks(tick_marks, iris.target_names, rotation=45)
    # plt.yticks(tick_marks, iris.target_names)

    tick_marks = np.arange(1, 11)
    plt.xticks(tick_marks - 0.5, tick_marks, fontsize=20)#, rotation=45)
    plt.yticks(tick_marks - 0.5, tick_marks, fontsize=20)




    ### 2nd Part
    plt.subplot(132)
    plt.pcolor(data2, cmap=cmap, vmin=0, vmax=676)
    # plt.pcolor(cm)

    plt.colorbar()
    plt.title('24 Components', fontsize=36)
    # plt.title('Confusion Matrix', fontsize=18)
    # plt.ylabel('|', fontsize=50)
    plt.xlabel('Predicted Subject', fontsize=30)

    # tick_marks = np.arange(len(iris.target_names))
    # plt.xticks(tick_marks, iris.target_names, rotation=45)
    # plt.yticks(tick_marks, iris.target_names)

    tick_marks = np.arange(1, 11)
    plt.xticks(tick_marks - 0.5, tick_marks, fontsize=20)  # , rotation=45)
    plt.yticks(tick_marks - 0.5, tick_marks, fontsize=20)



    ### 3rd Part
    plt.subplot(133)
    plt.pcolor(data3, cmap=cmap, vmin=0, vmax=676)
    # plt.pcolor(cm)

    plt.colorbar()
    plt.title('50 Components', fontsize=36)
    # plt.title('Confusion Matrix', fontsize=18)
    # plt.ylabel('||', fontsize=50)
    plt.xlabel('Predicted Subject', fontsize=30)

    # tick_marks = np.arange(len(iris.target_names))
    # plt.xticks(tick_marks, iris.target_names, rotation=45)
    # plt.yticks(tick_marks, iris.target_names)
    tick_marks = np.arange(1, 11)
    plt.xticks(tick_marks - 0.5, tick_marks, fontsize=20)  # , rotation=45)
    plt.yticks(tick_marks - 0.5, tick_marks, fontsize=20)

    plt.tight_layout()

    return plt



def plot_heatmap(Data_container, cmap=plt.cm.GnBu, vmin=0., vmax=1.):
    ### Plotting Section
    plt.style.use('ggplot')

    plt.figure(num=None, figsize=[16, 8])
    plt.subplot(111)
    plt.subplots_adjust(top=0.96, bottom=0.11, right=1.1, left=0.055)

    Data_container = np.ma.masked_invalid(Data_container)

    # plt.pcolor(Data_container, cmap=plt.cm.GnBu, vmin=0, vmax=1)
    plt.pcolor(Data_container, cmap=cmap, vmin=vmin, vmax=vmax)
    # plt.pcolor(Data_container, cmap=plt.cm.bwr, vmin=vmin, vmax=vmax)
    plt.colorbar()

    # plt.title('Confusion Matrix', fontsize=18)
    plt.ylabel('Elevation Angle', fontsize=16)
    plt.xlabel('Azimuthal Angle', fontsize=16)

    # tick_marks = np.arange(len(iris.target_names))
    # plt.xticks(tick_marks, iris.target_names, rotation=45)
    # plt.yticks(tick_marks, iris.target_names)

    x_tick_marks = np.linspace(-130, 135, 54)
    # x_tick_marks = np.linspace(-130, 135, 27)
    y_tick_marks = np.linspace(-40, 95, 28)
    plt.xticks(np.arange(1, 54) - 0.5, x_tick_marks, rotation=45)  # , fontsize=12)#, rotation=45)
    plt.yticks(np.arange(1, 28) - 0.5, y_tick_marks)  # , fontsize=12)
    plt.xlim([0, 53])
    plt.ylim([0, 27])

    plt.vlines(26.5, 0, 27, color='black', alpha=0.3)
    plt.vlines(35.5, 0, 27, color='black', alpha=0.3)
    plt.vlines(44.5, 0, 27, color='black', alpha=0.3)
    plt.vlines(17.5, 0, 27, color='black', alpha=0.3)
    plt.vlines(8.5, 0, 27, color='black', alpha=0.3)

    plt.hlines(8.5, 0, 53, color='black', alpha=0.3)
    plt.hlines(17.5, 0, 53, color='black', alpha=0.3)

    return plt



def plot_heatmap_diff(Data_container1, Data_container2, cmap=plt.cm.GnBu, vmin=0., vmax=1.):
    ### Plotting Section
    plt.style.use('ggplot')

    plt.figure(num=None, figsize=[28, 8])
    plt.subplot(131)
    plt.subplots_adjust(top=0.94, bottom=0.11, right=1, left=0.04)

    Data_container1 = np.ma.masked_invalid(Data_container1)

    # plt.pcolor(Data_container1, cmap=plt.cm.GnBu, vmin=0, vmax=1)
    plt.pcolor(Data_container1, cmap=cmap, vmin=vmin, vmax=vmax)
    # plt.pcolor(Data_container1, cmap=plt.cm.bwr, vmin=vmin, vmax=vmax)
    plt.colorbar()

    plt.title('PCA', fontsize=36)
    plt.ylabel('Elevation Angle', fontsize=30)
    plt.xlabel('Azimuthal Angle', fontsize=30)

    # tick_marks = np.arange(len(iris.target_names))
    # plt.xticks(tick_marks, iris.target_names, rotation=45)
    # plt.yticks(tick_marks, iris.target_names)

    # x_tick_marks = np.linspace(-130, 135, 54)
    x_tick_marks = ['', '', '', '', '', '', '', '', -90,
                    '', '', '', '', '', '', '', '', -45,
                    '', '', '', '', '', '', '', '', 0,
                    '', '', '', '', '', '', '', '', 45,
                    '', '', '', '', '', '', '', '', 90,
                    '', '', '', '', '', '', '', '']
    # x_tick_marks = np.linspace(-130, 135, 27)
    # y_tick_marks = np.linspace(-40, 95, 28)
    y_tick_marks = [-40, '', '', '', '', '', '', '', 0,
                    '', '', '', '', '', '', '', '', 45,
                    '', '', '', '', '', '', '', '', 90]
    plt.xticks(np.arange(1, 54) - 0.5, x_tick_marks, fontsize=20)#, rotation=45)  # , fontsize=12)#, rotation=45)
    plt.yticks(np.arange(1, 28) - 0.5, y_tick_marks, fontsize=20)  # , fontsize=12)
    plt.xlim([0, 53])
    plt.ylim([0, 27])

    plt.vlines(26.5, 0, 27, color='black', alpha=0.3)
    plt.vlines(35.5, 0, 27, color='black', alpha=0.3)
    plt.vlines(44.5, 0, 27, color='black', alpha=0.3)
    plt.vlines(17.5, 0, 27, color='black', alpha=0.3)
    plt.vlines(8.5, 0, 27, color='black', alpha=0.3)

    plt.hlines(8.5, 0, 53, color='black', alpha=0.3)
    plt.hlines(17.5, 0, 53, color='black', alpha=0.3)



    ### Second Part

    plt.subplot(132)
    Data_container2 = np.ma.masked_invalid(Data_container2)

    # plt.pcolor(Data_container, cmap=plt.cm.GnBu, vmin=0, vmax=1)
    plt.pcolor(Data_container2, cmap=cmap, vmin=vmin, vmax=vmax)
    # plt.pcolor(Data_container, cmap=plt.cm.bwr, vmin=vmin, vmax=vmax)
    plt.colorbar()

    # plt.title('Regularized NMF', fontsize=36)
    plt.title('NMF', fontsize=36)
    # plt.ylabel('Elevation Angle', fontsize=16)
    plt.ylabel('|', fontsize=50)
    plt.xlabel('Azimuthal Angle', fontsize=30)

    # tick_marks = np.arange(len(iris.target_names))
    # plt.xticks(tick_marks, iris.target_names, rotation=45)
    # plt.yticks(tick_marks, iris.target_names)

    # x_tick_marks = np.linspace(-130, 135, 54)
    x_tick_marks = ['', '', '', '', '', '', '', '', -90,
                    '', '', '', '', '', '', '', '', -45,
                    '', '', '', '', '', '', '', '', 0,
                    '', '', '', '', '', '', '', '', 45,
                    '', '', '', '', '', '', '', '', 90,
                    '', '', '', '', '', '', '', '']
    # x_tick_marks = np.linspace(-130, 135, 27)
    # y_tick_marks = np.linspace(-40, 95, 28)
    y_tick_marks = [-40, '', '', '', '', '', '', '', 0,
                    '', '', '', '', '', '', '', '', 45,
                    '', '', '', '', '', '', '', '', 90]
    plt.xticks(np.arange(1, 54) - 0.5, x_tick_marks, fontsize=20)#, rotation=45)  # , fontsize=12)#, rotation=45)
    plt.yticks(np.arange(1, 28) - 0.5, y_tick_marks, fontsize=20)  # , fontsize=12)
    plt.xlim([0, 53])
    plt.ylim([0, 27])

    plt.vlines(26.5, 0, 27, color='black', alpha=0.3)
    plt.vlines(35.5, 0, 27, color='black', alpha=0.3)
    plt.vlines(44.5, 0, 27, color='black', alpha=0.3)
    plt.vlines(17.5, 0, 27, color='black', alpha=0.3)
    plt.vlines(8.5, 0, 27, color='black', alpha=0.3)

    plt.hlines(8.5, 0, 53, color='black', alpha=0.3)
    plt.hlines(17.5, 0, 53, color='black', alpha=0.3)



    ### Third Part

    plt.subplot(133)
    Data_container3 = np.ma.masked_invalid(Data_container1 - Data_container2)

    # plt.pcolor(Data_container, cmap=plt.cm.GnBu, vmin=0, vmax=1)
    # plt.pcolor(Data_container3, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.pcolor(Data_container3, cmap=plt.cm.bwr, vmin=-0.5, vmax=0.5)
    plt.colorbar()

    plt.title('Difference', fontsize=36)
    # plt.ylabel('Elevation Angle', fontsize=16)
    plt.ylabel('||', fontsize=50)
    plt.xlabel('Azimuthal Angle', fontsize=30)

    # tick_marks = np.arange(len(iris.target_names))
    # plt.xticks(tick_marks, iris.target_names, rotation=45)
    # plt.yticks(tick_marks, iris.target_names)

    # x_tick_marks = np.linspace(-130, 135, 54)
    x_tick_marks = ['', '', '', '', '', '', '', '', -90,
                    '', '', '', '', '', '', '', '', -45,
                    '', '', '', '', '', '', '', '', 0,
                    '', '', '', '', '', '', '', '', 45,
                    '', '', '', '', '', '', '', '', 90,
                    '', '', '', '', '', '', '', '']
    # x_tick_marks = np.linspace(-130, 135, 27)
    # y_tick_marks = np.linspace(-40, 95, 28)
    y_tick_marks = [-40, '', '', '', '', '', '', '', 0,
                    '', '', '', '', '', '', '', '', 45,
                    '', '', '', '', '', '', '', '', 90]
    plt.xticks(np.arange(1, 54) - 0.5, x_tick_marks, fontsize=20)#, rotation=45)  # , fontsize=12)#, rotation=45)
    plt.yticks(np.arange(1, 28) - 0.5, y_tick_marks, fontsize=20)  # , fontsize=12)
    plt.xlim([0, 53])
    plt.ylim([0, 27])

    plt.vlines(26.5, 0, 27, color='black', alpha=0.3)
    plt.vlines(35.5, 0, 27, color='black', alpha=0.3)
    plt.vlines(44.5, 0, 27, color='black', alpha=0.3)
    plt.vlines(17.5, 0, 27, color='black', alpha=0.3)
    plt.vlines(8.5, 0, 27, color='black', alpha=0.3)

    plt.hlines(8.5, 0, 53, color='black', alpha=0.3)
    plt.hlines(17.5, 0, 53, color='black', alpha=0.3)

    plt.tight_layout()

    return plt


def plot_heatmap_diff1(Data_container1, Data_container2, cmap=plt.cm.GnBu, vmin=0., vmax=1.):
    ### Plotting Section
    plt.style.use('ggplot')

    plt.figure(num=None, figsize=[28, 8])
    plt.subplot(131)
    plt.subplots_adjust(top=0.94, bottom=0.11, right=1, left=0.04)

    Data_container1 = np.ma.masked_invalid(Data_container1)

    # plt.pcolor(Data_container1, cmap=plt.cm.GnBu, vmin=0, vmax=1)
    plt.pcolor(Data_container1, cmap=cmap, vmin=vmin, vmax=vmax)
    # plt.pcolor(Data_container1, cmap=plt.cm.bwr, vmin=vmin, vmax=vmax)
    plt.colorbar()

    plt.title('NMF', fontsize=36)
    plt.ylabel('Elevation Angle', fontsize=30)
    plt.xlabel('Azimuthal Angle', fontsize=30)

    # tick_marks = np.arange(len(iris.target_names))
    # plt.xticks(tick_marks, iris.target_names, rotation=45)
    # plt.yticks(tick_marks, iris.target_names)

    # x_tick_marks = np.linspace(-130, 135, 54)
    x_tick_marks = ['', '', '', '', '', '', '', '', -90,
                    '', '', '', '', '', '', '', '', -45,
                    '', '', '', '', '', '', '', '', 0,
                    '', '', '', '', '', '', '', '', 45,
                    '', '', '', '', '', '', '', '', 90,
                    '', '', '', '', '', '', '', '']
    # x_tick_marks = np.linspace(-130, 135, 27)
    # y_tick_marks = np.linspace(-40, 95, 28)
    y_tick_marks = [-40, '', '', '', '', '', '', '', 0,
                    '', '', '', '', '', '', '', '', 45,
                    '', '', '', '', '', '', '', '', 90]
    plt.xticks(np.arange(1, 54) - 0.5, x_tick_marks, fontsize=20)#, rotation=45)  # , fontsize=12)#, rotation=45)
    plt.yticks(np.arange(1, 28) - 0.5, y_tick_marks, fontsize=20)  # , fontsize=12)
    plt.xlim([0, 53])
    plt.ylim([0, 27])

    plt.vlines(26.5, 0, 27, color='black', alpha=0.3)
    plt.vlines(35.5, 0, 27, color='black', alpha=0.3)
    plt.vlines(44.5, 0, 27, color='black', alpha=0.3)
    plt.vlines(17.5, 0, 27, color='black', alpha=0.3)
    plt.vlines(8.5, 0, 27, color='black', alpha=0.3)

    plt.hlines(8.5, 0, 53, color='black', alpha=0.3)
    plt.hlines(17.5, 0, 53, color='black', alpha=0.3)



    ### Second Part

    plt.subplot(132)
    Data_container2 = np.ma.masked_invalid(Data_container2)

    # plt.pcolor(Data_container, cmap=plt.cm.GnBu, vmin=0, vmax=1)
    plt.pcolor(Data_container2, cmap=cmap, vmin=vmin, vmax=vmax)
    # plt.pcolor(Data_container, cmap=plt.cm.bwr, vmin=vmin, vmax=vmax)
    plt.colorbar()

    plt.title('Regularized NMF', fontsize=36)
    # plt.ylabel('Elevation Angle', fontsize=16)
    plt.ylabel('|', fontsize=50)
    plt.xlabel('Azimuthal Angle', fontsize=30)

    # tick_marks = np.arange(len(iris.target_names))
    # plt.xticks(tick_marks, iris.target_names, rotation=45)
    # plt.yticks(tick_marks, iris.target_names)

    # x_tick_marks = np.linspace(-130, 135, 54)
    x_tick_marks = ['', '', '', '', '', '', '', '', -90,
                    '', '', '', '', '', '', '', '', -45,
                    '', '', '', '', '', '', '', '', 0,
                    '', '', '', '', '', '', '', '', 45,
                    '', '', '', '', '', '', '', '', 90,
                    '', '', '', '', '', '', '', '']
    # x_tick_marks = np.linspace(-130, 135, 27)
    # y_tick_marks = np.linspace(-40, 95, 28)
    y_tick_marks = [-40, '', '', '', '', '', '', '', 0,
                    '', '', '', '', '', '', '', '', 45,
                    '', '', '', '', '', '', '', '', 90]
    plt.xticks(np.arange(1, 54) - 0.5, x_tick_marks, fontsize=20)#, rotation=45)  # , fontsize=12)#, rotation=45)
    plt.yticks(np.arange(1, 28) - 0.5, y_tick_marks, fontsize=20)  # , fontsize=12)
    plt.xlim([0, 53])
    plt.ylim([0, 27])

    plt.vlines(26.5, 0, 27, color='black', alpha=0.3)
    plt.vlines(35.5, 0, 27, color='black', alpha=0.3)
    plt.vlines(44.5, 0, 27, color='black', alpha=0.3)
    plt.vlines(17.5, 0, 27, color='black', alpha=0.3)
    plt.vlines(8.5, 0, 27, color='black', alpha=0.3)

    plt.hlines(8.5, 0, 53, color='black', alpha=0.3)
    plt.hlines(17.5, 0, 53, color='black', alpha=0.3)



    ### Third Part

    plt.subplot(133)
    Data_container3 = np.ma.masked_invalid(Data_container1 - Data_container2)

    # plt.pcolor(Data_container, cmap=plt.cm.GnBu, vmin=0, vmax=1)
    # plt.pcolor(Data_container3, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.pcolor(Data_container3, cmap=plt.cm.bwr, vmin=-0.5, vmax=0.5)
    plt.colorbar()

    plt.title('Difference', fontsize=36)
    # plt.ylabel('Elevation Angle', fontsize=16)
    plt.ylabel('||', fontsize=50)
    plt.xlabel('Azimuthal Angle', fontsize=30)

    # tick_marks = np.arange(len(iris.target_names))
    # plt.xticks(tick_marks, iris.target_names, rotation=45)
    # plt.yticks(tick_marks, iris.target_names)

    # x_tick_marks = np.linspace(-130, 135, 54)
    x_tick_marks = ['', '', '', '', '', '', '', '', -90,
                    '', '', '', '', '', '', '', '', -45,
                    '', '', '', '', '', '', '', '', 0,
                    '', '', '', '', '', '', '', '', 45,
                    '', '', '', '', '', '', '', '', 90,
                    '', '', '', '', '', '', '', '']
    # x_tick_marks = np.linspace(-130, 135, 27)
    # y_tick_marks = np.linspace(-40, 95, 28)
    y_tick_marks = [-40, '', '', '', '', '', '', '', 0,
                    '', '', '', '', '', '', '', '', 45,
                    '', '', '', '', '', '', '', '', 90]
    plt.xticks(np.arange(1, 54) - 0.5, x_tick_marks, fontsize=20)#, rotation=45)  # , fontsize=12)#, rotation=45)
    plt.yticks(np.arange(1, 28) - 0.5, y_tick_marks, fontsize=20)  # , fontsize=12)
    plt.xlim([0, 53])
    plt.ylim([0, 27])

    plt.vlines(26.5, 0, 27, color='black', alpha=0.3)
    plt.vlines(35.5, 0, 27, color='black', alpha=0.3)
    plt.vlines(44.5, 0, 27, color='black', alpha=0.3)
    plt.vlines(17.5, 0, 27, color='black', alpha=0.3)
    plt.vlines(8.5, 0, 27, color='black', alpha=0.3)

    plt.hlines(8.5, 0, 53, color='black', alpha=0.3)
    plt.hlines(17.5, 0, 53, color='black', alpha=0.3)

    plt.tight_layout()

    return plt


def confusion_mat_egf_unnorl():
    egf_comp = [6, 10, 18, 50, 100]

    for egf_comp in egf_comp:
        y_true = np.load('../CLFs/SVM/predictions/egf_unnorl_%d_comp_true_label.npy' % egf_comp)
        y_pred = np.load('../CLFs/SVM/predictions/egf_unnorl_%d_comp_pred.npy' % egf_comp)

        con_mat = None

        for i in range(13):

            if con_mat is None:
                con_mat = confusion_matrix(y_true[i], y_pred[i])
            else:
                con_mat = con_mat + confusion_matrix(y_true[i], y_pred[i])

                # print con_mat
                # print np.sum(con_mat, axis=1)

                # plot_heatmap1(con_mat, num=i)

        plot_conf_mat(con_mat / float(np.sum(con_mat[0, :])), num=1)

        plt.savefig('../FIGs/plot_ave_nor_conf_mat_egf_unnorl_%d_comp.pdf' % egf_comp, format='pdf')
    # plt.show()


def confusion_mat_egf_norl():
    egf_comp = [8, 14, 24, 50, 100]

    for egf_comp in egf_comp:
        y_true = np.load('../CLFs/SVM/predictions/egf_norl_%d_comp_true_label.npy' % egf_comp)
        y_pred = np.load('../CLFs/SVM/predictions/egf_norl_%d_comp_pred.npy' % egf_comp)

        con_mat = None

        for i in range(13):

            if con_mat is None:
                con_mat = confusion_matrix(y_true[i], y_pred[i])
            else:
                con_mat = con_mat + confusion_matrix(y_true[i], y_pred[i])

                # print con_mat
                # print np.sum(con_mat, axis=1)

                # plot_heatmap1(con_mat, num=i)

        plot_conf_mat(con_mat / float(np.sum(con_mat[0, :])), num=1)

        plt.savefig('../FIGs/plot_ave_nor_conf_mat_egf_norl_%d_comp.pdf' % egf_comp, format='pdf')
    # plt.show()


def confusion_mat_egf_norl_r3():
    egf_comp = [8, 14, 24, 50, 100]

    for egf_comp in egf_comp:
        y_true = np.load('../CLFs/SVM/predictions/egf_norl_r3_%d_comp_true_label.npy' % egf_comp)
        y_pred = np.load('../CLFs/SVM/predictions/egf_norl_r3_%d_comp_pred.npy' % egf_comp)

        con_mat = None

        for i in range(13):

            if con_mat is None:
                con_mat = confusion_matrix(y_true[i], y_pred[i])
            else:
                con_mat = con_mat + confusion_matrix(y_true[i], y_pred[i])

                # print con_mat
                # print np.sum(con_mat, axis=1)

                # plot_heatmap1(con_mat, num=i)

        plot_conf_mat(con_mat / float(np.sum(con_mat[0, :])), num=1)

        plt.savefig('../FIGs/plot_ave_nor_conf_mat_egf_norl_r3_%d_comp.pdf' % egf_comp, format='pdf')
    # plt.show()


def confusion_mat_nmf():
    nmf_comp = [2, 8, 14, 24, 50]

    for nmf_comp in nmf_comp:
        y_true = np.load('../CLFs/SVM/predictions/nmf_reg_%d_comp_true_label.npy' % nmf_comp)
        y_pred = np.load('../CLFs/SVM/predictions/nmf_reg_%d_comp_pred.npy' % nmf_comp)

        con_mat = None

        for i in range(13):

            if con_mat is None:
                con_mat = confusion_matrix(y_true[i], y_pred[i])
            else:
                con_mat = con_mat + confusion_matrix(y_true[i], y_pred[i])

                # print con_mat
                # print np.sum(con_mat, axis=1)

                # plot_heatmap1(con_mat, num=i)

        plot_conf_mat(con_mat / float(np.sum(con_mat[0, :])), num=1)

        # plt.savefig('../FIGs/plot_ave_nor_conf_mat_nmf_%d_comp.pdf' % nmf_comp, format='pdf')
    plt.show()


def confusion_mat_diff_of_best():
    egf_y_true = np.load('../CLFs/SVM/predictions/egf_norl_%d_comp_true_label.npy' % 50)
    egf_y_pred = np.load('../CLFs/SVM/predictions/egf_norl_%d_comp_pred.npy' % 50)

    nmf_y_true = np.load('../CLFs/SVM/predictions/nmf_%d_comp_true_label.npy' % 24)
    nmf_y_pred = np.load('../CLFs/SVM/predictions/nmf_%d_comp_pred.npy' % 24)

    egf_con_mat = None

    for i in range(13):

        if egf_con_mat is None:
            egf_con_mat = confusion_matrix(egf_y_true[i], egf_y_pred[i])
        else:
            egf_con_mat = egf_con_mat + confusion_matrix(egf_y_true[i], egf_y_pred[i])

    nmf_con_mat = None

    for i in range(13):

        if nmf_con_mat is None:
            nmf_con_mat = confusion_matrix(nmf_y_true[i], nmf_y_pred[i])
        else:
            nmf_con_mat = nmf_con_mat + confusion_matrix(nmf_y_true[i], nmf_y_pred[i])

    # diff_con_mat = egf_con_mat / float(np.sum(egf_con_mat[0, :])) - nmf_con_mat / float(np.sum(nmf_con_mat[0, :]))
    # diff_con_mat = egf_con_mat - nmf_con_mat

    plt = plot_conf_mat_diff(egf_con_mat, nmf_con_mat, num=1)

    # plt.savefig('../FIGs/plot_ave_nor_conf_mat_diff.pdf', format='pdf')
    plt.savefig('../FIGs/DIFF_ConfusionMat_EF_NMF.pdf', format='pdf')
    plt.show()


def confusion_mat_diff_of_best1():
    # egf_y_true = np.load('../CLFs/SVM/predictions/egf_norl_%d_comp_true_label.npy' % 50)
    # egf_y_pred = np.load('../CLFs/SVM/predictions/egf_norl_%d_comp_pred.npy' % 50)

    egf_y_true = np.load('../CLFs/SVM/predictions/nmf_%d_comp_true_label.npy' % 50)
    egf_y_pred = np.load('../CLFs/SVM/predictions/nmf_%d_comp_pred.npy' % 50)

    nmf_y_true = np.load('../CLFs/SVM/predictions/nmf_reg_%d_comp_true_label.npy' % 24)
    nmf_y_pred = np.load('../CLFs/SVM/predictions/nmf_reg_%d_comp_pred.npy' % 24)

    egf_con_mat = None

    for i in range(13):

        if egf_con_mat is None:
            egf_con_mat = confusion_matrix(egf_y_true[i], egf_y_pred[i])
        else:
            egf_con_mat = egf_con_mat + confusion_matrix(egf_y_true[i], egf_y_pred[i])

    nmf_con_mat = None

    for i in range(13):

        if nmf_con_mat is None:
            nmf_con_mat = confusion_matrix(nmf_y_true[i], nmf_y_pred[i])
        else:
            nmf_con_mat = nmf_con_mat + confusion_matrix(nmf_y_true[i], nmf_y_pred[i])

    # diff_con_mat = egf_con_mat / float(np.sum(egf_con_mat[0, :])) - nmf_con_mat / float(np.sum(nmf_con_mat[0, :]))
    # diff_con_mat = egf_con_mat - nmf_con_mat

    plt = plot_conf_mat_diff(egf_con_mat, nmf_con_mat, num=1)

    # plt.savefig('../FIGs/plot_ave_nor_conf_mat_diff.pdf', format='pdf')
    plt.savefig('../FIGs/DIFF_ConfusionMat_NMF_NMF_Reg.pdf', format='pdf')
    plt.show()


def confusion_mat_egf_triple():
    egf_y_true1 = np.load('../CLFs/SVM/predictions/egf_norl_%d_comp_true_label.npy' % 8)
    egf_y_pred1 = np.load('../CLFs/SVM/predictions/egf_norl_%d_comp_pred.npy' % 8)

    egf_y_true2 = np.load('../CLFs/SVM/predictions/egf_norl_%d_comp_true_label.npy' % 24)
    egf_y_pred2 = np.load('../CLFs/SVM/predictions/egf_norl_%d_comp_pred.npy' % 24)

    egf_y_true3 = np.load('../CLFs/SVM/predictions/egf_norl_%d_comp_true_label.npy' % 50)
    egf_y_pred3 = np.load('../CLFs/SVM/predictions/egf_norl_%d_comp_pred.npy' % 50)


    egf_con_mat1 = None

    for i in range(13):

        if egf_con_mat1 is None:
            egf_con_mat1 = confusion_matrix(egf_y_true1[i], egf_y_pred1[i])
        else:
            egf_con_mat1 = egf_con_mat1 + confusion_matrix(egf_y_true1[i], egf_y_pred1[i])

    egf_con_mat2 = None

    for i in range(13):

        if egf_con_mat2 is None:
            egf_con_mat2 = confusion_matrix(egf_y_true2[i], egf_y_pred2[i])
        else:
            egf_con_mat2 = egf_con_mat2 + confusion_matrix(egf_y_true2[i], egf_y_pred2[i])

    egf_con_mat3 = None

    for i in range(13):

        if egf_con_mat3 is None:
            egf_con_mat3 = confusion_matrix(egf_y_true3[i], egf_y_pred3[i])
        else:
            egf_con_mat3 = egf_con_mat3 + confusion_matrix(egf_y_true3[i], egf_y_pred3[i])

    # diff_con_mat = egf_con_mat1 / float(np.sum(egf_con_mat1[0, :])) - nmf_con_mat / float(np.sum(nmf_con_mat[0, :]))
    # diff_con_mat = egf_con_mat1 - nmf_con_mat

    plt = plot_conf_mat_triple(egf_con_mat1, egf_con_mat2, egf_con_mat3, num=1)

    # plt.savefig('../FIGs/plot_ave_nor_conf_mat_diff.pdf', format='pdf')
    plt.savefig('../FIGs/Triple_ConfusionMat.pdf', format='pdf')
    plt.show()


def confusion_mat_nmf_triple1():
    egf_y_true1 = np.load('../CLFs/SVM/predictions/nmf_14_comp_true_label.npy')
    egf_y_pred1 = np.load('../CLFs/SVM/predictions/nmf_14_comp_pred.npy')

    egf_y_true2 = np.load('../CLFs/SVM/predictions/nmf_24_comp_true_label.npy')
    egf_y_pred2 = np.load('../CLFs/SVM/predictions/nmf_24_comp_pred.npy')

    egf_y_true3 = np.load('../CLFs/SVM/predictions/nmf_50_comp_true_label.npy')
    egf_y_pred3 = np.load('../CLFs/SVM/predictions/nmf_50_comp_pred.npy')


    egf_con_mat1 = None

    for i in range(13):

        if egf_con_mat1 is None:
            egf_con_mat1 = confusion_matrix(egf_y_true1[i], egf_y_pred1[i])
        else:
            egf_con_mat1 = egf_con_mat1 + confusion_matrix(egf_y_true1[i], egf_y_pred1[i])

    egf_con_mat2 = None

    for i in range(13):

        if egf_con_mat2 is None:
            egf_con_mat2 = confusion_matrix(egf_y_true2[i], egf_y_pred2[i])
        else:
            egf_con_mat2 = egf_con_mat2 + confusion_matrix(egf_y_true2[i], egf_y_pred2[i])

    egf_con_mat3 = None

    for i in range(13):

        if egf_con_mat3 is None:
            egf_con_mat3 = confusion_matrix(egf_y_true3[i], egf_y_pred3[i])
        else:
            egf_con_mat3 = egf_con_mat3 + confusion_matrix(egf_y_true3[i], egf_y_pred3[i])

    # diff_con_mat = egf_con_mat1 / float(np.sum(egf_con_mat1[0, :])) - nmf_con_mat / float(np.sum(nmf_con_mat[0, :]))
    # diff_con_mat = egf_con_mat1 - nmf_con_mat

    plt = plot_conf_mat_triple(egf_con_mat1, egf_con_mat2, egf_con_mat3, num=1)

    # plt.savefig('../FIGs/plot_ave_nor_conf_mat_diff.pdf', format='pdf')
    plt.savefig('../FIGs/Triple_ConfusionMat_NMF.pdf', format='pdf')
    plt.show()


def confusion_mat_nmf_reg_triple2():
    egf_y_true1 = np.load('../CLFs/SVM/predictions/nmf_reg_14_comp_true_label.npy')
    egf_y_pred1 = np.load('../CLFs/SVM/predictions/nmf_reg_14_comp_pred.npy')

    egf_y_true2 = np.load('../CLFs/SVM/predictions/nmf_reg_24_comp_true_label.npy')
    egf_y_pred2 = np.load('../CLFs/SVM/predictions/nmf_reg_24_comp_pred.npy')

    egf_y_true3 = np.load('../CLFs/SVM/predictions/nmf_reg_50_comp_true_label.npy')
    egf_y_pred3 = np.load('../CLFs/SVM/predictions/nmf_reg_50_comp_pred.npy')

    egf_con_mat1 = None

    for i in range(13):

        if egf_con_mat1 is None:
            egf_con_mat1 = confusion_matrix(egf_y_true1[i], egf_y_pred1[i])
        else:
            egf_con_mat1 = egf_con_mat1 + confusion_matrix(egf_y_true1[i], egf_y_pred1[i])

    egf_con_mat2 = None

    for i in range(13):

        if egf_con_mat2 is None:
            egf_con_mat2 = confusion_matrix(egf_y_true2[i], egf_y_pred2[i])
        else:
            egf_con_mat2 = egf_con_mat2 + confusion_matrix(egf_y_true2[i], egf_y_pred2[i])

    egf_con_mat3 = None

    for i in range(13):

        if egf_con_mat3 is None:
            egf_con_mat3 = confusion_matrix(egf_y_true3[i], egf_y_pred3[i])
        else:
            egf_con_mat3 = egf_con_mat3 + confusion_matrix(egf_y_true3[i], egf_y_pred3[i])

    # diff_con_mat = egf_con_mat1 / float(np.sum(egf_con_mat1[0, :])) - nmf_con_mat / float(np.sum(nmf_con_mat[0, :]))
    # diff_con_mat = egf_con_mat1 - nmf_con_mat

    plt = plot_conf_mat_triple(egf_con_mat1, egf_con_mat2, egf_con_mat3, num=1)

    # plt.savefig('../FIGs/plot_ave_nor_conf_mat_diff.pdf', format='pdf')
    plt.savefig('../FIGs/Triple_ConfusionMat_NMF_Reg.pdf', format='pdf')
    plt.show()



def heatmap_zero_one_loss():
    with open('mapping_dict.json', 'r') as f:
        mapping_dict = json.load(f)
    print mapping_dict

    with open('TrainingIllums.txt') as f:
        train_Ill = [f.replace('\n', '') for f in f.readlines()]

    with open('TestingIllums.txt') as f:
        test_Ill = [f.replace('\n', '') for f in f.readlines()]

    print train_Ill
    print test_Ill

    Data_container = np.ndarray(shape=(12, 29), dtype=object)
    Data_container.fill(np.nan)
    # print Data_container

    # egf_comp = [6, 10, 20, 50, 100]
    egf_comp = [6]

    for egf_comp in egf_comp:
        y_true = np.load('../CLFs/SVM/predictions/egf_%d_comp_true_label.npy' % egf_comp)
        y_pred = np.load('../CLFs/SVM/predictions/egf_%d_comp_pred.npy' % egf_comp)

        for n_clf in range(12):
            for p in range(53):

                ### Getting index
                if p == 0:
                    index = train_Ill[n_clf]
                else:
                    index = test_Ill[p - 1]

                ind_pair = mapping_dict[index]

                if Data_container[ind_pair[0], ind_pair[1]] is np.nan:
                    Data_container[ind_pair[0], ind_pair[1]] = [[], []]

                for j in range(10):
                    Data_container[ind_pair[0], ind_pair[1]][0].append(y_pred[n_clf][p + 53 * j])
                    Data_container[ind_pair[0], ind_pair[1]][1].append(y_true[n_clf][p + 53 * j])
                    # print y_pred[n_clf][p + 53 * j]
                    # print y_true[n_clf][p + 53 * j]

    # print Data_container
    for i in range(12):
        for j in range(29):
            if Data_container[i, j] is not np.nan:
                # print zero_one_loss(Data_container[i, j][1], Data_container[i, j][0])
                Data_container[i, j] = zero_one_loss(Data_container[i, j][1], Data_container[i, j][0])
    Data_container = Data_container.astype(np.float)
    print Data_container

    Data_container[Data_container>=0] = 1

    ### Plotting Section
    plt.style.use('ggplot')

    plt.figure(num=1, figsize=[16, 8])
    plt.subplot(111)
    plt.subplots_adjust(top=0.96, bottom=0.08, right=1.1, left=0.05)

    Data_container = np.ma.masked_invalid(Data_container)

    # plt.pcolor(Data_container, cmap=plt.cm.GnBu, vmin=0, vmax=1)
    # plt.colorbar()

    plt.pcolor(Data_container, cmap=plt.cm.Greys, vmin=0, vmax=1)
    plt.colorbar()

    # plt.title('Confusion Matrix', fontsize=18)
    plt.ylabel('Elevation Angle', fontsize=16)
    plt.xlabel('Azimuthal Angle', fontsize=16)

    # tick_marks = np.arange(len(iris.target_names))
    # plt.xticks(tick_marks, iris.target_names, rotation=45)
    # plt.yticks(tick_marks, iris.target_names)

    x_tick_marks = [-130, -120, -110, -95, -85, -70, -60, -50, -35, -25, -20, -15, -10, -5, 0, 5, 10,
                    15, 20, 25, 35, 50, 60, 70, 85, 95, 110, 120, 130]
    y_tick_marks = [-40, -35, -20, -10, 0, 10, 15, 20, 40, 45, 65, 90]
    plt.xticks(np.arange(1, 30) - 0.5, x_tick_marks, fontsize=12)#, rotation=45)
    plt.yticks(np.arange(1, 13) - 0.5, y_tick_marks, fontsize=12)
    plt.xlim([0, 29])
    plt.show()


def heatmap_zero_one_loss_rbb():
    with open('mapping_dict.json', 'r') as f:
        mapping_dict = json.load(f)
    print mapping_dict

    with open('TrainingIllums.txt') as f:
        train_Ill = [f.replace('\n', '') for f in f.readlines()]

    with open('TestingIllums.txt') as f:
        test_Ill = [f.replace('\n', '') for f in f.readlines()]

    print train_Ill
    print test_Ill

    x_range = [-130, -120, -110, -95, -85, -70, -60, -50, -35, -25, -20, -15, -10, -5, 0, 5, 10,
               15, 20, 25, 35, 50, 60, 70, 85, 95, 110, 120, 130]
    y_range = [-40, -35, -20, -10, 0, 10, 15, 20, 40, 45, 65, 90]

    # nmf_comp = [2, 8, 14, 24, 50]
    nmf_comp = [2]

    for nmf_comp in nmf_comp:
        y_true = np.load('../CLFs/SVM/predictions/nmf_%d_comp_true_label.npy' % nmf_comp)
        y_pred = np.load('../CLFs/SVM/predictions/nmf_%d_comp_pred.npy' % nmf_comp)

        Data_container = np.ndarray(shape=(27, 53), dtype=object)
        Data_container.fill(np.nan)
        # print Data_container


        for n_clf in range(13):
            for p in range(52):

                ### Getting index
                if p == 0:
                    index = train_Ill[n_clf]
                else:
                    index = test_Ill[p - 1]

                ind_pair = mapping_dict[index]

                ind_pair = [(y_range[ind_pair[0]] + 40) / 5, (x_range[ind_pair[1]] + 130) / 5]

                # print 'ind_pair', ind_pair
                if Data_container[ind_pair[0], ind_pair[1]] is np.nan:
                    Data_container[ind_pair[0], ind_pair[1]] = [[], []]

                for j in range(10):
                    Data_container[ind_pair[0], ind_pair[1]][0].append(y_pred[n_clf][p + 52 * j])
                    Data_container[ind_pair[0], ind_pair[1]][1].append(y_true[n_clf][p + 52 * j])
                    # print y_pred[n_clf][p + 53 * j]
                    # print y_true[n_clf][p + 53 * j]

        # print Data_container
        for i in range(27):
            for j in range(53):
                if Data_container[i, j] is not np.nan:
                    # print zero_one_loss(Data_container[i, j][1], Data_container[i, j][0])
                    Data_container[i, j] = zero_one_loss(Data_container[i, j][1], Data_container[i, j][0])
        Data_container = Data_container.astype(np.float)
        # print Data_container

        Data_container[Data_container>=0] = 1

        ### Plotting Section
        plt.style.use('ggplot')

        plt.figure(num=None, figsize=[16, 8])
        plt.subplot(111)
        plt.subplots_adjust(top=0.96, bottom=0.11, right=1.1, left=0.055)

        Data_container = np.ma.masked_invalid(Data_container)

        plt.pcolor(Data_container, cmap=plt.cm.Greys, vmin=0, vmax=1)
        plt.colorbar()

        # plt.title('Confusion Matrix', fontsize=18)
        plt.ylabel('Elevation Angle', fontsize=16)
        plt.xlabel('Azimuthal Angle', fontsize=16)

        # tick_marks = np.arange(len(iris.target_names))
        # plt.xticks(tick_marks, iris.target_names, rotation=45)
        # plt.yticks(tick_marks, iris.target_names)

        x_tick_marks = np.linspace(-130, 135, 54)
        # x_tick_marks = np.linspace(-130, 135, 27)
        y_tick_marks = np.linspace(-40, 95, 28)
        plt.xticks(np.arange(1, 54) - 0.5, x_tick_marks, rotation=45)  # , fontsize=12)#, rotation=45)
        plt.yticks(np.arange(1, 28) - 0.5, y_tick_marks)  # , fontsize=12)
        plt.xlim([0, 53])
        plt.ylim([0, 27])

        plt.vlines(26.5, 0, 27, color='black', alpha=0.3)
        plt.vlines(35.5, 0, 27, color='black', alpha=0.3)
        plt.vlines(44.5, 0, 27, color='black', alpha=0.3)
        plt.vlines(17.5, 0, 27, color='black', alpha=0.3)
        plt.vlines(8.5, 0, 27, color='black', alpha=0.3)

        plt.hlines(8.5, 0, 53, color='black', alpha=0.3)
        plt.hlines(17.5, 0, 53, color='black', alpha=0.3)

        plt.savefig('../FIGs/plot_heatmap_zero_one_loss_black.pdf', format='pdf')
    plt.show()


def heatmap_zero_one_loss_egf_unnorl():
    with open('mapping_dict.json', 'r') as f:
        mapping_dict = json.load(f)
    # print mapping_dict

    with open('TrainingIllums.txt') as f:
        train_Ill = [f.replace('\n', '') for f in f.readlines()]

    with open('TestingIllums.txt') as f:
        test_Ill = [f.replace('\n', '') for f in f.readlines()]

    # print train_Ill
    # print test_Ill

    x_range = [-130, -120, -110, -95, -85, -70, -60, -50, -35, -25, -20, -15, -10, -5, 0, 5, 10,
                    15, 20, 25, 35, 50, 60, 70, 85, 95, 110, 120, 130]
    y_range = [-40, -35, -20, -10, 0, 10, 15, 20, 40, 45, 65, 90]

    egf_comp = [6, 10, 18, 50, 100]
    # egf_comp = [6]
    plt = None
    for egf_comp in egf_comp:
        y_true = np.load('../CLFs/SVM/predictions/egf_unnorl_%d_comp_true_label.npy' % egf_comp)
        y_pred = np.load('../CLFs/SVM/predictions/egf_unnorl_%d_comp_pred.npy' % egf_comp)

        Data_container = np.ndarray(shape=(27, 53), dtype=object)
        Data_container.fill(np.nan)
        # print Data_container


        for n_clf in range(13):
            for p in range(52):

                ### Getting index
                if p == 0:
                    index = train_Ill[n_clf]
                else:
                    index = test_Ill[p - 1]

                ind_pair = mapping_dict[index]

                ind_pair = [(y_range[ind_pair[0]]+40)/5, (x_range[ind_pair[1]]+130)/5]

                # print 'ind_pair', ind_pair
                if Data_container[ind_pair[0], ind_pair[1]] is np.nan:
                    Data_container[ind_pair[0], ind_pair[1]] = [[], []]

                for j in range(10):
                    Data_container[ind_pair[0], ind_pair[1]][0].append(y_pred[n_clf][p + 52 * j])
                    Data_container[ind_pair[0], ind_pair[1]][1].append(y_true[n_clf][p + 52 * j])
                    # print y_pred[n_clf][p + 53 * j]
                    # print y_true[n_clf][p + 53 * j]

        # print Data_container
        for i in range(27):
            for j in range(53):
                if Data_container[i, j] is not np.nan:
                    # print zero_one_loss(Data_container[i, j][1], Data_container[i, j][0])
                    Data_container[i, j] = zero_one_loss(Data_container[i, j][1], Data_container[i, j][0])
        Data_container = Data_container.astype(np.float)
        # print Data_container

        # Data_container[Data_container>=0] = 0.8

        ### Plotting Section

        plt = plot_heatmap(Data_container)

        plt.savefig('../FIGs/plot_heatmap_zero_one_loss_egf_unnorl_%d_comp.pdf' % egf_comp, format='pdf')
    # plt.show()


def heatmap_zero_one_loss_egf_norl():
    with open('mapping_dict.json', 'r') as f:
        mapping_dict = json.load(f)
    # print mapping_dict

    with open('TrainingIllums.txt') as f:
        train_Ill = [f.replace('\n', '') for f in f.readlines()]

    with open('TestingIllums.txt') as f:
        test_Ill = [f.replace('\n', '') for f in f.readlines()]

    # print train_Ill
    # print test_Ill

    x_range = [-130, -120, -110, -95, -85, -70, -60, -50, -35, -25, -20, -15, -10, -5, 0, 5, 10,
               15, 20, 25, 35, 50, 60, 70, 85, 95, 110, 120, 130]
    y_range = [-40, -35, -20, -10, 0, 10, 15, 20, 40, 45, 65, 90]

    mean_list = []
    std_list = []

    egf_comp = [8, 14, 24, 50, 100]
    # egf_comp = [6]
    plt = None
    for egf_comp in egf_comp:
        y_true = np.load('../CLFs/SVM/predictions/egf_norl_%d_comp_true_label.npy' % egf_comp)
        y_pred = np.load('../CLFs/SVM/predictions/egf_norl_%d_comp_pred.npy' % egf_comp)

        Data_container = np.ndarray(shape=(27, 53), dtype=object)
        Data_container.fill(np.nan)
        # print Data_container


        for n_clf in range(13):
            for p in range(52):

                ### Getting index
                if p == 0:
                    index = train_Ill[n_clf]
                else:
                    index = test_Ill[p - 1]

                ind_pair = mapping_dict[index]

                ind_pair = [(y_range[ind_pair[0]] + 40) / 5, (x_range[ind_pair[1]] + 130) / 5]

                # print 'ind_pair', ind_pair
                if Data_container[ind_pair[0], ind_pair[1]] is np.nan:
                    Data_container[ind_pair[0], ind_pair[1]] = [[], []]

                for j in range(10):
                    Data_container[ind_pair[0], ind_pair[1]][0].append(y_pred[n_clf][p + 52 * j])
                    Data_container[ind_pair[0], ind_pair[1]][1].append(y_true[n_clf][p + 52 * j])
                    # print y_pred[n_clf][p + 53 * j]
                    # print y_true[n_clf][p + 53 * j]

        # print Data_container
        for i in range(27):
            for j in range(53):
                if Data_container[i, j] is not np.nan:
                    # print zero_one_loss(Data_container[i, j][1], Data_container[i, j][0])
                    Data_container[i, j] = zero_one_loss(Data_container[i, j][1], Data_container[i, j][0])
        Data_container = Data_container.astype(np.float)
        # print Data_container

        ### Mis-classification rate
        mean_list.append(np.nanmean(Data_container))
        std_list.append(np.nanstd(Data_container))

        # Data_container[Data_container>=0] = 0.8

        ### Plotting Section

        # plt = plot_heatmap(Data_container)
        # plt.savefig('../FIGs/plot_heatmap_zero_one_loss_egf_norl_%d_comp.pdf' % egf_comp, format='pdf')

    print 'egf_norl'
    print mean_list
    print std_list
    # plt.show()


def heatmap_zero_one_loss_egf_norl_r3():
    with open('mapping_dict.json', 'r') as f:
        mapping_dict = json.load(f)
    # print mapping_dict

    with open('TrainingIllums.txt') as f:
        train_Ill = [f.replace('\n', '') for f in f.readlines()]

    with open('TestingIllums.txt') as f:
        test_Ill = [f.replace('\n', '') for f in f.readlines()]

    # print train_Ill
    # print test_Ill

    x_range = [-130, -120, -110, -95, -85, -70, -60, -50, -35, -25, -20, -15, -10, -5, 0, 5, 10,
               15, 20, 25, 35, 50, 60, 70, 85, 95, 110, 120, 130]
    y_range = [-40, -35, -20, -10, 0, 10, 15, 20, 40, 45, 65, 90]

    mean_list = []
    std_list = []

    egf_comp = [8, 14, 24, 50, 100]
    # egf_comp = [6]

    plt = None
    for egf_comp in egf_comp:
        y_true = np.load('../CLFs/SVM/predictions/egf_norl_r3_%d_comp_true_label.npy' % egf_comp)
        y_pred = np.load('../CLFs/SVM/predictions/egf_norl_r3_%d_comp_pred.npy' % egf_comp)

        Data_container = np.ndarray(shape=(27, 53), dtype=object)
        Data_container.fill(np.nan)
        # print Data_container


        for n_clf in range(13):
            for p in range(52):

                ### Getting index
                if p == 0:
                    index = train_Ill[n_clf]
                else:
                    index = test_Ill[p - 1]

                ind_pair = mapping_dict[index]

                ind_pair = [(y_range[ind_pair[0]] + 40) / 5, (x_range[ind_pair[1]] + 130) / 5]

                # print 'ind_pair', ind_pair
                if Data_container[ind_pair[0], ind_pair[1]] is np.nan:
                    Data_container[ind_pair[0], ind_pair[1]] = [[], []]

                for j in range(10):
                    Data_container[ind_pair[0], ind_pair[1]][0].append(y_pred[n_clf][p + 52 * j])
                    Data_container[ind_pair[0], ind_pair[1]][1].append(y_true[n_clf][p + 52 * j])
                    # print y_pred[n_clf][p + 53 * j]
                    # print y_true[n_clf][p + 53 * j]

        # print Data_container
        for i in range(27):
            for j in range(53):
                if Data_container[i, j] is not np.nan:
                    # print zero_one_loss(Data_container[i, j][1], Data_container[i, j][0])
                    Data_container[i, j] = zero_one_loss(Data_container[i, j][1], Data_container[i, j][0])
        Data_container = Data_container.astype(np.float)
        # print Data_container

        ### Mis-classification rate
        mean_list.append(np.nanmean(Data_container))
        std_list.append(np.nanstd(Data_container))

        # Data_container[Data_container>=0] = 0.8

        # plt = plot_heatmap(Data_container)
        # plt.savefig('../FIGs/plot_heatmap_zero_one_loss_egf_norl_r3_%d_comp.pdf' % egf_comp, format='pdf')

    print 'egf_norl_r3'
    print mean_list
    print std_list
    # plt.show()


def heatmap_zero_one_loss_nmf():
    with open('mapping_dict.json', 'r') as f:
        mapping_dict = json.load(f)
    # print mapping_dict

    with open('TrainingIllums.txt') as f:
        train_Ill = [f.replace('\n', '') for f in f.readlines()]

    with open('TestingIllums.txt') as f:
        test_Ill = [f.replace('\n', '') for f in f.readlines()]

    # print train_Ill
    # print test_Ill

    x_range = [-130, -120, -110, -95, -85, -70, -60, -50, -35, -25, -20, -15, -10, -5, 0, 5, 10,
               15, 20, 25, 35, 50, 60, 70, 85, 95, 110, 120, 130]
    y_range = [-40, -35, -20, -10, 0, 10, 15, 20, 40, 45, 65, 90]

    mean_list = []
    std_list = []

    nmf_comp = [2, 8, 14, 24, 50]
    # nmf_comp = [6]
    plt = None
    for nmf_comp in nmf_comp:
        y_true = np.load('../CLFs/SVM/predictions/nmf_reg_%d_comp_true_label.npy' % nmf_comp)
        y_pred = np.load('../CLFs/SVM/predictions/nmf_reg_%d_comp_pred.npy' % nmf_comp)

        Data_container = np.ndarray(shape=(27, 53), dtype=object)
        Data_container.fill(np.nan)
        # print Data_container


        for n_clf in range(13):
            for p in range(52):

                ### Getting index
                if p == 0:
                    index = train_Ill[n_clf]
                else:
                    index = test_Ill[p - 1]

                ind_pair = mapping_dict[index]

                ind_pair = [(y_range[ind_pair[0]] + 40) / 5, (x_range[ind_pair[1]] + 130) / 5]

                # print 'ind_pair', ind_pair
                if Data_container[ind_pair[0], ind_pair[1]] is np.nan:
                    Data_container[ind_pair[0], ind_pair[1]] = [[], []]

                for j in range(10):
                    Data_container[ind_pair[0], ind_pair[1]][0].append(y_pred[n_clf][p + 52 * j])
                    Data_container[ind_pair[0], ind_pair[1]][1].append(y_true[n_clf][p + 52 * j])
                    # print y_pred[n_clf][p + 53 * j]
                    # print y_true[n_clf][p + 53 * j]

        # print Data_container
        for i in range(27):
            for j in range(53):
                if Data_container[i, j] is not np.nan:
                    # print zero_one_loss(Data_container[i, j][1], Data_container[i, j][0])
                    Data_container[i, j] = zero_one_loss(Data_container[i, j][1], Data_container[i, j][0])
        Data_container = Data_container.astype(np.float)

        ### Mis-classification rate
        # print np.isnan(Data_container)
        mean_list.append(np.nanmean(Data_container))
        std_list.append(np.nanstd(Data_container))

        # Data_container[Data_container>=0] = 0.8


        ### Plotting Section
        # plt = plot_heatmap(Data_container)
        # plt.savefig('../FIGs/plot_heatmap_zero_one_loss_nmf_%d_comp.pdf' % nmf_comp, format='pdf')


    print 'NMF_reg'
    print mean_list
    print std_list
    # plt.show()


def heatmap_zero_one_loss_diff():
    with open('mapping_dict.json', 'r') as f:
        mapping_dict = json.load(f)
    # print mapping_dict

    with open('TrainingIllums.txt') as f:
        train_Ill = [f.replace('\n', '') for f in f.readlines()]

    with open('TestingIllums.txt') as f:
        test_Ill = [f.replace('\n', '') for f in f.readlines()]

    # print train_Ill
    # print test_Ill

    x_range = [-130, -120, -110, -95, -85, -70, -60, -50, -35, -25, -20, -15, -10, -5, 0, 5, 10,
               15, 20, 25, 35, 50, 60, 70, 85, 95, 110, 120, 130]
    y_range = [-40, -35, -20, -10, 0, 10, 15, 20, 40, 45, 65, 90]

    mean_list = []
    std_list = []

    # nmf_comp = [2, 8, 14, 24, 50]
    # nmf_comp = [6]
    plt = None
    # for nmf_comp in nmf_comp:
    nmf_y_true = np.load('../CLFs/SVM/predictions/nmf_reg_24_comp_true_label.npy')
    nmf_y_pred = np.load('../CLFs/SVM/predictions/nmf_reg_24_comp_pred.npy')

    egf_y_true = np.load('../CLFs/SVM/predictions/egf_norl_50_comp_true_label.npy')
    egf_y_pred = np.load('../CLFs/SVM/predictions/egf_norl_50_comp_pred.npy')

    nmf_Data_container = np.ndarray(shape=(27, 53), dtype=object)
    nmf_Data_container.fill(np.nan)

    egf_Data_container = np.ndarray(shape=(27, 53), dtype=object)
    egf_Data_container.fill(np.nan)
    # print nmf_Data_container


    for n_clf in range(13):
        for p in range(52):

            ### Getting index
            if p == 0:
                index = train_Ill[n_clf]
            else:
                index = test_Ill[p - 1]

            ind_pair = mapping_dict[index]

            ind_pair = [(y_range[ind_pair[0]] + 40) / 5, (x_range[ind_pair[1]] + 130) / 5]

            # print 'ind_pair', ind_pair
            if nmf_Data_container[ind_pair[0], ind_pair[1]] is np.nan:
                nmf_Data_container[ind_pair[0], ind_pair[1]] = [[], []]

            if egf_Data_container[ind_pair[0], ind_pair[1]] is np.nan:
                egf_Data_container[ind_pair[0], ind_pair[1]] = [[], []]

            for j in range(10):
                nmf_Data_container[ind_pair[0], ind_pair[1]][0].append(nmf_y_pred[n_clf][p + 52 * j])
                nmf_Data_container[ind_pair[0], ind_pair[1]][1].append(nmf_y_true[n_clf][p + 52 * j])

                egf_Data_container[ind_pair[0], ind_pair[1]][0].append(egf_y_pred[n_clf][p + 52 * j])
                egf_Data_container[ind_pair[0], ind_pair[1]][1].append(egf_y_true[n_clf][p + 52 * j])
                # print nmf_y_pred[n_clf][p + 53 * j]
                # print nmf_y_true[n_clf][p + 53 * j]

    # print nmf_Data_container
    for i in range(27):
        for j in range(53):
            if nmf_Data_container[i, j] is not np.nan:
                # print zero_one_loss(nmf_Data_container[i, j][1], nmf_Data_container[i, j][0])
                nmf_Data_container[i, j] = zero_one_loss(nmf_Data_container[i, j][1], nmf_Data_container[i, j][0])
    nmf_Data_container = nmf_Data_container.astype(np.float)

    for i in range(27):
        for j in range(53):
            if egf_Data_container[i, j] is not np.nan:
                # print zero_one_loss(nmf_Data_container[i, j][1], nmf_Data_container[i, j][0])
                egf_Data_container[i, j] = zero_one_loss(egf_Data_container[i, j][1], egf_Data_container[i, j][0])
    egf_Data_container = egf_Data_container.astype(np.float)

    ### Mis-classification rate
    # print np.isnan(nmf_Data_container)
    # mean_list.append(np.nanmean(nmf_Data_container))
    # std_list.append(np.nanstd(nmf_Data_container))

    # nmf_Data_container[nmf_Data_container>=0] = 0.8

    # egf_Data_container[egf_Data_container == 0] = 0.0001
    # nmf_Data_container[nmf_Data_container == 0] = 0.0001

    ### Plotting Section
    # plt = plot_heatmap((egf_Data_container - nmf_Data_container), vmin=-0.5, vmax=0.5)
    plt = plot_heatmap_diff(egf_Data_container, nmf_Data_container)
    # plt.savefig('../FIGs/plot_heatmap_zero_one_loss_egf_nmf_diff.pdf', format='pdf')
    plt.savefig('../FIGs/DIFF_Heatmap_EF_NMF_Reg.pdf', format='pdf')

    # print 'NMF'
    # print mean_list
    # print std_list
    plt.show()



def heatmap_zero_one_loss_diff1():
    with open('mapping_dict.json', 'r') as f:
        mapping_dict = json.load(f)
    # print mapping_dict

    with open('TrainingIllums.txt') as f:
        train_Ill = [f.replace('\n', '') for f in f.readlines()]

    with open('TestingIllums.txt') as f:
        test_Ill = [f.replace('\n', '') for f in f.readlines()]

    # print train_Ill
    # print test_Ill

    x_range = [-130, -120, -110, -95, -85, -70, -60, -50, -35, -25, -20, -15, -10, -5, 0, 5, 10,
               15, 20, 25, 35, 50, 60, 70, 85, 95, 110, 120, 130]
    y_range = [-40, -35, -20, -10, 0, 10, 15, 20, 40, 45, 65, 90]

    mean_list = []
    std_list = []

    # nmf_comp = [2, 8, 14, 24, 50]
    # nmf_comp = [6]
    plt = None
    # for nmf_comp in nmf_comp:
    nmf_y_true = np.load('../CLFs/SVM/predictions/nmf_24_comp_true_label.npy')
    nmf_y_pred = np.load('../CLFs/SVM/predictions/nmf_24_comp_pred.npy')

    nmf_reg_y_true = np.load('../CLFs/SVM/predictions/nmf_reg_24_comp_true_label.npy')
    nmf_reg_y_pred = np.load('../CLFs/SVM/predictions/nmf_reg_24_comp_pred.npy')

    nmf_Data_container = np.ndarray(shape=(27, 53), dtype=object)
    nmf_Data_container.fill(np.nan)

    nmf_reg_Data_container = np.ndarray(shape=(27, 53), dtype=object)
    nmf_reg_Data_container.fill(np.nan)
    # print nmf_Data_container


    for n_clf in range(13):
        for p in range(52):

            ### Getting index
            if p == 0:
                index = train_Ill[n_clf]
            else:
                index = test_Ill[p - 1]

            ind_pair = mapping_dict[index]

            ind_pair = [(y_range[ind_pair[0]] + 40) / 5, (x_range[ind_pair[1]] + 130) / 5]

            # print 'ind_pair', ind_pair
            if nmf_Data_container[ind_pair[0], ind_pair[1]] is np.nan:
                nmf_Data_container[ind_pair[0], ind_pair[1]] = [[], []]

            if nmf_reg_Data_container[ind_pair[0], ind_pair[1]] is np.nan:
                nmf_reg_Data_container[ind_pair[0], ind_pair[1]] = [[], []]

            for j in range(10):
                nmf_Data_container[ind_pair[0], ind_pair[1]][0].append(nmf_y_pred[n_clf][p + 52 * j])
                nmf_Data_container[ind_pair[0], ind_pair[1]][1].append(nmf_y_true[n_clf][p + 52 * j])

                nmf_reg_Data_container[ind_pair[0], ind_pair[1]][0].append(nmf_reg_y_pred[n_clf][p + 52 * j])
                nmf_reg_Data_container[ind_pair[0], ind_pair[1]][1].append(nmf_reg_y_true[n_clf][p + 52 * j])
                # print nmf_y_pred[n_clf][p + 53 * j]
                # print nmf_y_true[n_clf][p + 53 * j]

    # print nmf_Data_container
    for i in range(27):
        for j in range(53):
            if nmf_Data_container[i, j] is not np.nan:
                # print zero_one_loss(nmf_Data_container[i, j][1], nmf_Data_container[i, j][0])
                nmf_Data_container[i, j] = zero_one_loss(nmf_Data_container[i, j][1], nmf_Data_container[i, j][0])
    nmf_Data_container = nmf_Data_container.astype(np.float)

    for i in range(27):
        for j in range(53):
            if nmf_reg_Data_container[i, j] is not np.nan:
                # print zero_one_loss(nmf_Data_container[i, j][1], nmf_Data_container[i, j][0])
                nmf_reg_Data_container[i, j] = zero_one_loss(nmf_reg_Data_container[i, j][1], nmf_reg_Data_container[i, j][0])
    nmf_reg_Data_container = nmf_reg_Data_container.astype(np.float)

    ### Mis-classification rate
    # print np.isnan(nmf_Data_container)
    # mean_list.append(np.nanmean(nmf_Data_container))
    # std_list.append(np.nanstd(nmf_Data_container))

    # nmf_Data_container[nmf_Data_container>=0] = 0.8

    # nmf_reg_Data_container[nmf_reg_Data_container == 0] = 0.0001
    # nmf_Data_container[nmf_Data_container == 0] = 0.0001

    ### Plotting Section
    # plt = plot_heatmap((nmf_reg_Data_container - nmf_Data_container), vmin=-0.5, vmax=0.5)
    plt = plot_heatmap_diff1(nmf_Data_container, nmf_reg_Data_container)
    # plt.savefig('../FIGs/plot_heatmap_zero_one_loss_egf_nmf_diff.pdf', format='pdf')
    plt.savefig('../FIGs/DIFF_Heatmap_NMF_NMF_Reg.pdf', format='pdf')

    # print 'NMF'
    # print mean_list
    # print std_list
    plt.show()


def plot_line_with_r3():
    egf_norl_mean = [0.36286057692307694, 0.37271634615384619, 0.35504807692307694, 0.34903846153846158, 0.54350961538461529]
    egf_norl_std = \
        np.array([0.35692005834137808, 0.36899514138126643, 0.35785826058770531, 0.38305814558550805, 0.40733660723526299]) / np.sqrt(53)
    egf_norl_r3_mean = [0.46778846153846154, 0.45264423076923077, 0.46117788461538461, 0.40504807692307693, 0.59471153846153846]
    egf_norl_r3_std = \
        np.array([0.36480438494506606, 0.39691212611142268, 0.39339834651093414, 0.39576207932071317, 0.39475779734552269]) / np.sqrt(53)

    NMF_mean = [0.81262019230769234, 0.44230769230769229, 0.39002403846153844, 0.36165865384615387, 0.36959134615384615]
    NMF_std = \
        np.array([0.10651283352111494, 0.37365636999820662, 0.36681319837088122, 0.36661102928909917, 0.35734689093044503]) / np.sqrt(53)


    plt.style.use('ggplot')

    plt.figure(num=None, figsize=[10, 8])

    plt.subplot(211)
    plt.subplots_adjust(top=0.96, bottom=0.07, right=0.97, left=0.07)

    # plt.plot([8, 14, 24, 50, 100], egf_norl_mean, '-', linewidth=1.5, zorder=1, label='PCA')
    # plt.plot([8, 14, 24, 50, 100], egf_norl_r3_mean, '-', linewidth=1.5, zorder=1, label='PCA-3Comp')
    plt.scatter([8, 14, 24, 50, 100], egf_norl_mean, s=14, zorder=2, color='#3A0DDC', label='PCA')
    plt.scatter([8, 14, 24, 50, 100], egf_norl_r3_mean, s=14, zorder=2, color='#DC0D26', label='PCA (w/out first 3)')
    plt.errorbar([8, 14, 24, 50, 100], egf_norl_mean, yerr=egf_norl_std, zorder=1, linewidth=1.5, color='#3A0DDC', label='SEM')
    plt.errorbar([8, 14, 24, 50, 100], egf_norl_r3_mean, yerr=egf_norl_r3_std, zorder=1, linewidth=1.5, color='#DC0D26', label='SEM')

    plt.legend()
    plt.xlim([0, 110])
    plt.ylim([0, 1])
    plt.xlabel("Eigenfaces", size=12)
    plt.ylabel("Average misclassification rate", size=14)



    plt.subplot(212)
    # plt.plot([2, 8, 14, 24, 50], NMF_mean, '-', linewidth=1.5, zorder=1, label='PCA-3Comp')
    plt.scatter([2, 8, 14, 24, 50], NMF_mean, s=14, zorder=2, color='#00B339', label='NMF')
    plt.errorbar([2, 8, 14, 24, 50], NMF_mean, yerr=NMF_std, zorder=1, linewidth=1.5, color='#00B339', label='SEM')

    plt.legend()
    plt.xlim([0, 55])
    plt.ylim([0, 1])
    # ax1.set_xticks(list(range(0, 7)))
    # ax2.set_xticks(list(range(0, 7)))
    # ax1.set_xticklabels([8, 14, 24, 50, 100])
    # ax2.set_xticklabels([2, 8, 14, 24, 50])
    plt.xlabel("NMF components", size=12)
    plt.ylabel("Average misclassification rate", size=14)


    plt.savefig('../FIGs/plot_line_with_r3.pdf', format='pdf')
    plt.show()


def plot_line():
    egf_norl_mean = [0.36286057692307694, 0.37271634615384619, 0.35504807692307694, 0.34903846153846158, 0.54350961538461529]
    egf_norl_std = \
        np.array([0.35692005834137808, 0.36899514138126643, 0.35785826058770531, 0.38305814558550805, 0.40733660723526299]) / np.sqrt(53)
    egf_norl_r3_mean = [0.46778846153846154, 0.45264423076923077, 0.46117788461538461, 0.40504807692307693, 0.59471153846153846]
    egf_norl_r3_std = \
        np.array([0.36480438494506606, 0.39691212611142268, 0.39339834651093414, 0.39576207932071317, 0.39475779734552269]) / np.sqrt(53)

    NMF_mean = [0.81262019230769234, 0.44230769230769229, 0.39002403846153844, 0.36165865384615387, 0.36959134615384615]
    NMF_std = \
        np.array([0.10651283352111494, 0.37365636999820662, 0.36681319837088122, 0.36661102928909917, 0.35734689093044503]) / np.sqrt(53)

    NMF_reg_mean = [0.81298076923076912, 0.40805288461538458, 0.3828125, 0.32752403846153844, 0.3359375]
    NMF_reg_std = \
        np.array([0.11524731414293247, 0.36655034105653378, 0.38389774921922154, 0.36200819701108627, 0.36991636740595174]) / np.sqrt(53)

    plt.style.use('ggplot')

    fig = plt.figure(num=None, figsize=[10, 8])
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    plt.subplots_adjust(top=0.92, bottom=0.09, right=0.97, left=0.09)

    as1 = ax1.scatter([8, 14, 24, 50, 100], egf_norl_mean, s=14, zorder=2, color='#3A0DDC', label='PCA')
    ae1 = ax1.errorbar([8, 14, 24, 50, 100], egf_norl_mean, yerr=egf_norl_std, zorder=1, linewidth=1.5, color='#3A0DDC', label='SEM')

    # ax1.legend()
    ax1.set_xlim([0, 110])
    # ax1.ylim([0, 1])
    ax1.set_xlabel("Eigenfaces", size=20)
    ax1.set_ylabel("Average misclassification rate", size=20)

    as2 = ax2.scatter([2, 8, 14, 24, 50], NMF_mean, s=14, zorder=2, color='#EE0000', label='NMF')
    ae2 = ax2.errorbar([2, 8, 14, 24, 50], NMF_mean, yerr=NMF_std, zorder=1, linewidth=1.5, color='#EE0000',
                       label='SEM')
    as2 = ax2.scatter([2, 8, 14, 24, 50], NMF_reg_mean, s=14, zorder=2, color='#00B339', label='NMF_Reg')
    ae2 = ax2.errorbar([2, 8, 14, 24, 50], NMF_reg_mean, yerr=NMF_reg_std, zorder=1, linewidth=1.5, color='#00B339', label='SEM')

    # ax2.legend(loc=1)
    ax2.set_xlim([0, 55])
    ax2.set_ylim([0, 1])

    # ax1.set_xticks(list(range(0, 7)))
    # ax2.set_xticks(list(range(0, 7)))
    # ax1.set_xticklabels([8, 14, 24, 50, 100])
    # ax2.set_xticklabels([2, 8, 14, 24, 50])
    ax2.set_xlabel("NMF components", size=20)
    ax2.set_ylabel("Average misclassification rate", size=20)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0, fontsize=15)

    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    plt.savefig('../FIGs/plot_line_EF_NMF_NMF_Reg.pdf', format='pdf')
    plt.show()


def main():
    with open('mapping_dict.json', 'r') as f:
        mapping_dict = json.load(f)
    print mapping_dict

    with open('TrainingIllums.txt') as f:
        train_Ill = [f.replace('\n', '') for f in f.readlines()]

    with open('TestingIllums.txt') as f:
        test_Ill = [f.replace('\n', '') for f in f.readlines()]

    print train_Ill
    print test_Ill

    Data_container = np.ndarray(shape=(12, 29), dtype=object)
    Data_container.fill(None)
    # print Data_container

    # egf_comp = [6, 10, 20, 50, 100]
    egf_comp = [6]

    for egf_comp in egf_comp:
        y_true = np.load('../CLFs/SVM/predictions/egf_%d_comp_true_label.npy' % egf_comp)
        y_pred = np.load('../CLFs/SVM/predictions/egf_%d_comp_pred.npy' % egf_comp)

        for n_clf in range(12):
            for p in range(53):

                ### Getting index
                if p == 0:
                    index = train_Ill[n_clf]
                else:
                    index = test_Ill[p - 1]

                ind_pair = mapping_dict[index]

                if Data_container[ind_pair[0], ind_pair[1]] is None:
                    Data_container[ind_pair[0], ind_pair[1]] = [[], []]

                for j in range(10):
                    Data_container[ind_pair[0], ind_pair[1]][0].append(y_pred[n_clf][p + 53 * j])
                    Data_container[ind_pair[0], ind_pair[1]][1].append(y_true[n_clf][p + 53 * j])
                    # print y_pred[n_clf][p + 53 * j]
                    # print y_true[n_clf][p + 53 * j]

    # print Data_container
    for i in range(12):
        for j in range(29):
            if Data_container[i, j] is not None:
                # print zero_one_loss(Data_container[i, j][1], Data_container[i, j][0])
                Data_container[i, j] = zero_one_loss(Data_container[i, j][1], Data_container[i, j][0])

    print Data_container


if __name__ == '__main__':
    # main()

    # heatmap_zero_one_loss_rbb()

    ### EGF
    # confusion_mat_egf_unnorl()
    # confusion_mat_egf_norl()
    # confusion_mat_egf_norl_r3()

    # heatmap_zero_one_loss_egf_unnorl()
    # heatmap_zero_one_loss_egf_norl()
    # heatmap_zero_one_loss_egf_norl_r3()

    ### NMF
    # confusion_mat_nmf()
    # heatmap_zero_one_loss_nmf()

    # plot_line()
    # plot_line_with_r3()

    confusion_mat_diff_of_best()
    # heatmap_zero_one_loss_diff()
    # confusion_mat_egf_triple()

    ### NMF vs NFM_reg
    # confusion_mat_diff_of_best1()
    # heatmap_zero_one_loss_diff1()

    # confusion_mat_nmf_triple1()
    # confusion_mat_nmf_reg_triple2()

    ### Testing Code
    # y_true = [2, 0, 2, 2, 0, 1]
    # y_pred = [0, 0, 2, 2, 0, 2]
    # con_mat = confusion_matrix(y_true, y_pred)
    #
    # plot_heatmap1(con_mat)
