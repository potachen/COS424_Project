#!/usr/local/bin/python

import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import zero_one_loss


def plot_heatmap1(data, num=1, cmap=plt.cm.GnBu):
    plt.style.use('ggplot')

    plt.figure(num=None, figsize=[10, 8])
    plt.subplot(111)
    plt.subplots_adjust(top=0.96, bottom=0.08, right=1.05, left=0.08)

    plt.pcolor(data, cmap=cmap, vmin=0, vmax=1)
    # plt.pcolor(cm)

    plt.colorbar()
    # plt.title('Confusion Matrix', fontsize=18)
    plt.ylabel('True Subject', fontsize=16)
    plt.xlabel('Predicted Subject', fontsize=16)

    # tick_marks = np.arange(len(iris.target_names))
    # plt.xticks(tick_marks, iris.target_names, rotation=45)
    # plt.yticks(tick_marks, iris.target_names)

    tick_marks = np.arange(1, 11)
    plt.xticks(tick_marks - 0.5, tick_marks, rotation=45)
    plt.yticks(tick_marks - 0.5, tick_marks)

    # plt.savefig('../FIGs/plot_heatmap.pdf', format='pdf')
    # plt.savefig('../FIGs/plot_heatmap.png', format='png')

    # plt.show()


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

        plot_heatmap1(con_mat / float(np.sum(con_mat[0, :])), num=1)

        plt.savefig('../FIGs/plot_ave_nor_conf_mat_egf_unnor_%d_comp.pdf' % egf_comp, format='pdf')
    plt.show()


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

        plot_heatmap1(con_mat / float(np.sum(con_mat[0, :])), num=1)

        plt.savefig('../FIGs/plot_ave_nor_conf_mat_egf_nor_%d_comp.pdf' % egf_comp, format='pdf')
    plt.show()


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

        plot_heatmap1(con_mat / float(np.sum(con_mat[0, :])), num=1)

        plt.savefig('../FIGs/plot_ave_nor_conf_mat_egf_nor_%d_comp.pdf' % egf_comp, format='pdf')
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


def heatmap_zero_one_loss_egf_unnorl():
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

    egf_comp = [6, 10, 18, 50, 100]
    # egf_comp = [6]

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

                print 'ind_pair', ind_pair
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
        plt.style.use('ggplot')

        plt.figure(num=None, figsize=[16, 8])
        plt.subplot(111)
        plt.subplots_adjust(top=0.96, bottom=0.1, right=1.1, left=0.05)

        Data_container = np.ma.masked_invalid(Data_container)

        plt.pcolor(Data_container, cmap=plt.cm.GnBu, vmin=0, vmax=1)
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
        plt.xticks(np.arange(1, 54) - 0.5, x_tick_marks, rotation=45)#, fontsize=12)#, rotation=45)
        plt.yticks(np.arange(1, 28) - 0.5, y_tick_marks)#, fontsize=12)
        plt.xlim([0, 53])
        plt.ylim([0, 27])

        plt.savefig('../FIGs/plot_heatmap_zero_one_loss_egf_unnorl_%d_comp.pdf' % egf_comp, format='pdf')
    plt.show()


def heatmap_zero_one_loss_egf_norl():
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

    egf_comp = [8, 14, 24, 50, 100]
    # egf_comp = [6]

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

                print 'ind_pair', ind_pair
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
        plt.style.use('ggplot')

        plt.figure(num=None, figsize=[16, 8])
        plt.subplot(111)
        plt.subplots_adjust(top=0.96, bottom=0.1, right=1.1, left=0.05)

        Data_container = np.ma.masked_invalid(Data_container)

        plt.pcolor(Data_container, cmap=plt.cm.GnBu, vmin=0, vmax=1)
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

        plt.savefig('../FIGs/plot_heatmap_zero_one_loss_egf_norl_%d_comp.pdf' % egf_comp, format='pdf')
    plt.show()


def heatmap_zero_one_loss_egf_norl_r3():
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

    egf_comp = [8, 14, 24, 50, 100]
    # egf_comp = [6]

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

                print 'ind_pair', ind_pair
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
        plt.style.use('ggplot')

        plt.figure(num=None, figsize=[16, 8])
        plt.subplot(111)
        plt.subplots_adjust(top=0.96, bottom=0.1, right=1.1, left=0.05)

        Data_container = np.ma.masked_invalid(Data_container)

        plt.pcolor(Data_container, cmap=plt.cm.GnBu, vmin=0, vmax=1)
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

        plt.savefig('../FIGs/plot_heatmap_zero_one_loss_egf_norl_r3_%d_comp.pdf' % egf_comp, format='pdf')
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

    # heatmap_zero_one_loss1()

    # confusion_mat_egf_unnorl()
    # confusion_mat_egf_norl()
    # confusion_mat_egf_norl_r3()

    # heatmap_zero_one_loss_egf_unnorl()
    # heatmap_zero_one_loss_egf_norl()
    heatmap_zero_one_loss_egf_norl_r3()


    ### Testing Code
    # y_true = [2, 0, 2, 2, 0, 1]
    # y_pred = [0, 0, 2, 2, 0, 2]
    # con_mat = confusion_matrix(y_true, y_pred)
    #
    # plot_heatmap1(con_mat)
