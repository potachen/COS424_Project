#!/usr/local/bin/python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_heatmap1(data, title='Heatmap', cmap=plt.cm.Blues):

    plt.style.use('ggplot')

    plt.figure(num=None, figsize=[10, 8])
    plt.subplot(111)
    plt.subplots_adjust(top=0.94, bottom=0.08, right=1.05, left=0.08)

    plt.pcolor(data, cmap=cmap)
    # plt.pcolor(cm)

    plt.colorbar()
    plt.title(title, fontsize=18)
    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)

    # tick_marks = np.arange(len(iris.target_names))
    # plt.xticks(tick_marks, iris.target_names, rotation=45)
    # plt.yticks(tick_marks, iris.target_names)

    plt.savefig('../FIGs/plot_heatmap.pdf', format='pdf')

    plt.show()


def main():
    pass

if __name__ == '__main__':
    main()

    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]
    con_mat = confusion_matrix(y_true, y_pred)

    plot_heatmap1(con_mat)
