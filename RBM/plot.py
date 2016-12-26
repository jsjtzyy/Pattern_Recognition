# This code from scratch part is completed by ghe10, yingyiz2 and gjin7
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix_train(cm, classes, title='Confustion matrix of Training, logistic regression and PCA', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_confusion_matrix_eval(cm, classes, title='Confustion matrix of Testing, logistic regression and PCA', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')