# This code from scratch part is completed by yingyiz2, ghe10 and gjin7

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time

# use tensorflow API to load data from MNIST
def readData():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
    train = mnist.train.images
    return train


# define sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# define RBM class
class RBM:
    iteration = 20
    learning_rate = 0.1

    c = [] # bias at hidden layer
    b = [] # bias at visible layer
    W = [] # weight matrix

    dW = [] # delta W
    dc = [] # delta c
    db = [] # delta b

    dimh = 200 # the dimension of hidden layer
    dimv = 0   # the dimension of visible layer
    batch_size = 0 # batch size

    def __init__(self, iteration, data, batch_size):
        self.iteration = iteration
        self.batch_size = batch_size
        self.dimv = np.shape(data)[1]
        self.W = np.random.normal(0, 1.0, (self.dimh, self.dimv)) # h * v
        self.b = np.random.normal(-0.2, 0.2, (self.dimv, 1))
        self.c = np.random.normal(-0.2, 0.2, (self.dimh, 1))
        self.dW = np.zeros([self.dimh, self.dimv])
        self.db = np.zeros(self.dimv).reshape(self.dimv, 1)
        self.dc = np.zeros(self.dimh).reshape(self.dimh, 1)

    # use CD-k method for approximation in update equations
    def contrastive_divergence_k(self, data, k, offset):
        self.dW = np.zeros([self.dimh, self.dimv])
        self.db = np.zeros([self.dimv, 1])
        self.dc = np.zeros([self.dimh, 1])
        offset = offset * self.batch_size
        for i in range(self.batch_size):
            v = data[offset + i]
            v = np.array(v).reshape(784, 1)
            v0 = data[offset + i]
            v0 = np.array(v0).reshape(784, 1)
            for t in range(k):
                h = sigmoid(np.dot(self.W, v) + self.c)
                tmp = np.random.normal(0, 1, (self.dimh, 1))
                h = (np.sign(h - tmp) + 1) / 2   # sample

                v = sigmoid(np.dot(np.transpose(self.W), h) + self.b)
                tmp = np.random.normal(0, 1, (self.dimv, 1))
                v = (np.sign(v - tmp) + 1) / 2   # sample

            tmph_v0 = sigmoid(np.dot(self.W, v0) + self.c)
            tmph_vk = sigmoid(np.dot(self.W, v) + self.c)

            self.dW += np.dot(tmph_v0, v0.transpose()) - np.dot(tmph_vk, v.transpose())
            self.db += v0 - v
            self.dc += tmph_v0 - tmph_vk

    def train(self, data):
        total = np.shape(data)[0]
        batch_num = int(total / self.batch_size)
        for k in range(self.iteration):
            print("iteration: ", k)
            start = time.clock()
            for i in range(batch_num):
                self.contrastive_divergence_k(data, 1, i)
                self.W += self.learning_rate * self.dW / self.batch_size  # update weight matrix
                self.b += self.learning_rate * self.db / self.batch_size  # update bias vector b
                self.c += self.learning_rate * self.dc / self.batch_size  # update bias vector c
            end = time.clock()
            print("iteration running time is: ", end - start)

    # plot the 64 learned filters
    def plotFilter(self):
        for i in range(64):
            im = np.array(self.W[i])
            im = im.reshape(28, 28)
            ax = plt.subplot(8, 8, i+1)
            ax.imshow(im, cmap=plt.cm.gray)
            plt.axis('off')
        plt.show()


def __main__():
    data = readData()
    rbm = RBM(5, data, 500)
    rbm.train(data)
    rbm.plotFilter()

if __name__ == '__main__':
    __main__()





