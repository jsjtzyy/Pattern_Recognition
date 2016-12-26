# This code from scratch part is completed by ghe10, yingyiz2 and gjin7
import numpy as np
import time
import matplotlib.pyplot as plt
from read import readData
from plot import plot_confusion_matrix_train
from plot import plot_confusion_matrix_eval
from sklearn.metrics import confusion_matrix

def logistic(x):
    return 1 / (1 + np.exp(-x))


def diff_log(y):
    return y * (1 - y)


def relu(x):
    return (x > 0) * x


def diff_relu(y):
    return np.ones(np.shape(y)) * (y > 0)


def tanh(x):
    return np.tanh(x)


def diff_tanh(y):
    return 1.0 - y * y


def softmax(y):
    y_hat = np.exp(y)
    y_hat = y_hat / np.sum(y_hat)
    return y_hat


class NeuralNet:   # a simple neural network with two hidden layers
    layer_size = 10
    learning_rate = 0.1
    ''' network parameters '''
    W1 = []
    b1 = []
    W2 = []
    b2 = []
    W3 = []
    b3 = []

    ''' the following matrices are for gradient'''
    dW1 = []
    dW2 = []
    dW3 = []
    error = 0

    gradW3 = []
    gradW2 = []
    gradW1 = []

    y1 = []
    y2 = []
    y_out = []

    X = []
    Y = []

    def __init__(self, size, x, y, lr, activation, diff):
        self.layer_size = size
        self.learning_rate = lr
        self.in_size = 1120

        if activation == relu:
            print("relu")
            scale = 100
            thr = 0
        elif activation == tanh:
            print("tanh")
            scale = 1000
            thr = 0
        else:
            print("sigmoid")
            scale = 1
            thr = 0

        # initialization
        self.W1 = (np.random.normal(0, 0.2, (self.layer_size, self.in_size)) - thr) / scale
        self.gradW1 = np.zeros((self.layer_size, self.in_size))
        self.b1 = np.zeros((self.layer_size, 1))
        self.W2 = (np.random.normal(0, 0.2, (self.layer_size, self.layer_size)) - thr) / scale
        self.gradW2 = np.zeros((self.layer_size, self.layer_size))
        self.b2 = np.zeros((self.layer_size, 1))
        self.W3 = (np.random.normal(0, 0.2, (9, self.layer_size)) - thr) / scale
        self.gradW3 = (np.zeros((9, self.layer_size)) - thr) / scale
        self.b3 = np.zeros((9, 1))
        self.y1 = np.zeros((10, 1))
        self.y2 = np.zeros((10, 1))
        self.y_out = np.zeros((9, 1))
        self.X = x
        self.Y = y
        self.predict_y = []
        self.error = 0
        self.activation = activation
        self.diff = diff
        self.correct = 0    # compute correct number of points
        if activation == logistic:
            self.b1 = (np.random.randn(self.layer_size, 1)).reshape(self.layer_size, 1)
            self.b2 += (np.random.randn(self.layer_size, 1)).reshape(self.layer_size, 1)
            self.b3 += (np.random.randn(9, 1)).reshape(9, 1)

    def forward(self, x, y):
        '''
        tmp = np.dot(self.W1, x)
        tmp1 = tmp + self.b1 # after sum, 10 * 1 is converted to 10 * 10....
        tmp2 = self.activation(tmp)
        tmp3 = self.activation(tmp1)
        '''
        self.y1 = np.array(self.activation(np.dot(self.W1, x) + self.b1)).reshape(self.layer_size, 1)
        self.y2 = np.array(self.activation(np.dot(self.W2, self.y1) + self.b2)).reshape(self.layer_size, 1)
        self.y_out = np.array(logistic(np.dot(self.W3, self.y2) + self.b3)).reshape(9, 1)
        '''Compute the error'''
        self.error = -(y - self.y_out)

    def isCorrect(self, y):
        index1 = np.argmax(self.y_out)
        index2 = np.argmax(y)
        if index1 == index2:
            return True
        return False

    def backward(self, x):
        '''
        x : 1120 * 1
        W1 : 10 * 1120
        y1 : 10 * 1
        W2 : 10 * 10
        y2 : 10 * 1
        W3 : 9 * 10
        y3 : 9 * 1
        y : 9 * 1
        '''
        ''' note that the last layer is classifier, we use logistic for classifier'''
        self.dW3 = self.error * self.y_out * (1 - self.y_out)
        self.dW2 = np.dot(np.transpose(self.W3), self.dW3) * self.diff(self.y2)
        self.dW1 = np.dot(np.transpose(self.W2), self.dW2) * self.diff(self.y1)
        self.gradW3 += np.dot(self.dW3, np.transpose(self.y2))
        self.gradW2 += np.dot(self.dW2, np.transpose(self.y1))
        self.gradW1 += np.dot(self.dW1, np.transpose(x))

    def update(self):
        self.W1 = self.W1 - self.learning_rate * self.gradW1
        self.b1 = self.b1 - self.learning_rate * self.dW1
        self.W2 = self.W2 - self.learning_rate * self.gradW2
        self.b2 = self.b2 - self.learning_rate * self.dW2
        self.W3 = self.W3 - self.learning_rate * self.gradW3
        self.b3 = self.b3 - self.learning_rate * self.dW3

        self.gradW1 = np.zeros((self.layer_size, self.in_size))
        self.gradW2 = np.zeros((self.layer_size, self.layer_size))
        self.gradW3 = np.zeros((9, self.layer_size))

    def train(self, batch_size):
        size = np.shape(self.X)
        length = size[0]
        batches = int(length / batch_size)
        for i in range(batches):
            index = i * batch_size
            for j in range(batch_size):
                self.forward(self.X[index + j], self.Y[index + j])
                if self.isCorrect(self.Y[index + j]):
                    self.correct += 1
                self.backward(self.X[index + j])
            self.update()

    def test(self, X, Y, l, epoch):
        length = l
        count = 0
        wrong = 0
        for i in range(length):
            self.forward(X[i], Y[i])
            if self.isCorrect(Y[i]):
                count += 1
            else:
                wrong += 1
        print(epoch)
        print("accuracy : ", count / length)




def __main__():
    # epochs = 301
    # batch_size = 50
    # x, y, data_size,target = readData("train") # you can change data set here for "train", "dev" and "eval" three options
    # x_eval, y_eval, data_size_eval, target_eval = readData("eval")
    # net = NeuralNet(50, x, y, 0.01, logistic, diff_log)
    # for i in range(epochs):
    #     # start = time.clock()
    #     net.train(batch_size)
    #     # end = time.clock()
    #     # print ("One iteration running time is: ", end - start)
    #     if i % 10 == 0:
    #         net.test(x, y, data_size, i)
    #         net.test(x_eval, y_eval, data_size_eval, i)
    index = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
    # y_hat = []
    # for i in range(data_size):
    #     net.forward(x[i], y[i])
    #     y_hat.append(np.argmax(net.y_out))
    target = [0, 0, 0, 1, 1, 1, 1, 1, 1]
    y_hat = [0, 0, 0, 1, 1, 1, 1, 1, 1]
    cm = confusion_matrix(target, y_hat)
    plt.figure()
    plot_confusion_matrix_train(cm, index)
    plt.show()

    # y_hat = []
    # for i in range(data_size_eval):
    #     net.forward(x_eval[i], y_eval[i])
    #     y_hat.append(np.argmax(net.y_out))
    cm = confusion_matrix(target, y_hat)
    plt.figure()
    plot_confusion_matrix_eval(cm, index)
    plt.show()

if __name__ == '__main__':
    __main__()

