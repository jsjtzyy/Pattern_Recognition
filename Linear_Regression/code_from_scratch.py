# Co-author: 
# Guanchen He (NetID: ghe10)
# Guxin Jin (NetID: gjin7)
# Yingyi Zhang (NetID: yingyiz2)

import numpy as np
import matplotlib.pyplot as plt

import scipy.io as sio
import random

#state = 0 for logistic regression
#state = 1 for svm
#state = 2 for perceptron

state = 0



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def readData(folder, state):   # read data from the data set
    x = []
    y = []
    file = open("../hw1/" + folder + "/lab/hw1" + folder +"_labels.txt")
    cnt = 0
    for lines in file:
        words = lines.split()
        num = int(words[0])
        if num == 0 and state > 0:
            num = -1
        f = open("../hw1/" + words[1])
        raw_data = f.read()
        tmp_x = []
        isNAN = False
        tmp_x.append(1)
        for number in raw_data.split():
            if np.isnan(float(number)):
                isNAN = True            # remove the corrupt data
                break
            tmp_x.append(float(number))
        if isNAN == False:
            y.append(num)
            x.append(np.array(tmp_x))
            cnt += 1
        f.close()
    print("read finished")
    return x, y, cnt

def linear_regression(iteration):
    epochs = iteration
    learning_rate = 0.01            # set the learning rate
    X, Y, length = readData("train", 0)
    W = np.random.random((1, 17))
    W /= 100
    resX = []
    resY = []
    for epoch in range(epochs):
        correct = 0
        gradient = 0.0
        print("epoch is ", epoch)
        for i in range(length):
            yi = np.dot(W, X[i])
            gradient += ((yi - Y[i]) * X[i]) * 2 / length
            if yi >= 0.5 and Y[i] == 1:
                correct += 1
            elif yi < 0.5 and Y[i] == 0:
                correct += 1
        W -= learning_rate * gradient  #update the vector w
        resX.append(epoch + 1)
        resY.append(1 - correct / length)
    print(1 - correct / length)
    plt.subplot(2,2,1)
    plt.plot(resX, resY, '-r')


def logistic_regression(iteration):
    epochs = iteration
    learning_rate = 0.005
    X, Y, length = readData("train", 0)
    W = np.random.random((1, 17))
    W /= 100
    resX = []
    resY = []
    for epoch in range(epochs):
        correct = 0
        gradient = 0.0
        print("epoch is ", epoch)
        for i in range(length):
            yi = sigmoid(np.dot(W, X[i]))  # call logistic activation function
            gradient += ((yi - Y[i]) * yi * (1 - yi) * X[i]) * 2 / length
            if yi >= 0.5 and Y[i] == 1:
                correct += 1
            elif yi < 0.5 and Y[i] == 0:
                correct += 1
        W -= learning_rate * gradient
        resX.append(epoch + 1)
        resY.append(1 - correct / length)
    print(1 - correct / length)
    plt.subplot(2,2,2)
    plt.plot(resX, resY, '-y')

def perceptron(iteration):
    epochs = iteration
    learning_rate = 0.01
    X, Y, length = readData("train", 2)
    W = np.random.random((1, 17)) - 1
    W /= 1000
    resX = []
    resY = []
    for epoch in range(epochs):
        correct = 0
        gradient = 0.0
        print("epoch is ", epoch)
        for i in range(length):
            yi = np.dot(W, X[i])
            if (yi > 0 and Y[i] == -1) or (yi < 0 and Y[i] == 1):
                gradient +=  Y[i] * X[i] / length
            if yi > 0 and Y[i] == 1:
                correct += 1
            elif yi < 0 and Y[i] == -1:
                correct += 1
        W += learning_rate * gradient
        resX.append(epoch + 1)
        resY.append(1 - correct / length)
    print(1 - correct / length)
    plt.subplot(2,2,3)
    plt.plot(resX, resY, '-g')


def linear_svm(iteration):
    epochs = iteration
    learning_rate = 0.05
    X, Y, length = readData("train", 1)
    W = np.random.random((1, 17)) - 1
    W /= 1000
    C = 1
    lamda = 0.05
    resX = []
    resY = []
    for epoch in range(epochs):
        correct = 0
        gradient = 0.0
        print("epoch is ", epoch)
        for i in range(length):
            yi = np.dot(W, X[i])
            if yi * Y[i] < 1:
                gradient += Y[i] * X[i] / length
            if yi > 0 and Y[i] == 1:
                correct += 1
            elif yi < 0 and Y[i] == -1:
                correct += 1
        W -= learning_rate * (2 * lamda * W - C * gradient)
        resX.append(epoch + 1)
        resY.append(1 - correct / length)
    print(1 - correct / length)
    plt.subplot(2,2,4)
    plt.plot(resX, resY, '-b')

def __main__():
    steps = 500
    linear_svm(steps)
    perceptron(steps)
    logistic_regression(steps)
    linear_regression(steps)
    plt.show()
if __name__ == '__main__':
    __main__()
