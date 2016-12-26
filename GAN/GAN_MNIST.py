# Yingyi Zhang (yingyiz2) Date: 2016-12-1

'''
The code will download the MNIST Dataset automatically via tensorflow API
'''

import math
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import time


class GAN_MNIST():
    def __init__(self):
        self.learning_rate = 0.001
        self.noise_dim = 32
        self.g_hidden1_nodes = 64
        self.g_hidden2_nodes = 64
        self.image_dim = 784        # MNIST data set
        self.d_hidden1_nodes = 32
        self.d_hidden2_nodes = 32
        self.maxout_num = 5
        self.batch_size = 50
        self.iterations = 100
        self.batch_num = 55000 // self.batch_size
        self.X = tf.placeholder(tf.float32, shape=(None, self.image_dim))
        self.Z = tf.placeholder(tf.float32, shape=(None, self.noise_dim))
        self.keep_prob = tf.placeholder(tf.float32)
        self.generative_dim = (3, 4)

        with tf.variable_scope("G"):
            self.GW1 = tf.Variable(tf.random_normal([self.noise_dim, self.g_hidden1_nodes], stddev=0.1))
            self.Gb1 = tf.Variable(tf.zeros(self.g_hidden1_nodes))
            self.GW2 = tf.Variable(tf.random_normal([self.g_hidden1_nodes, self.g_hidden2_nodes], stddev=0.1))
            self.Gb2 = tf.Variable(tf.zeros(self.g_hidden2_nodes))
            self.GW3 = tf.Variable(tf.random_normal([self.g_hidden2_nodes, self.image_dim], stddev=0.1))
            self.Gb3 = tf.Variable(tf.zeros(self.image_dim))

        with tf.variable_scope("D"):
            self.DW1 = tf.Variable(tf.random_normal([self.image_dim, self.maxout_num * self.d_hidden1_nodes], stddev=0.01))
            self.Db1 = tf.Variable(tf.zeros(self.maxout_num * self.d_hidden1_nodes))
            self.DW2 = tf.Variable(tf.random_normal([self.d_hidden1_nodes, self.maxout_num * self.d_hidden2_nodes], stddev=0.01))
            self.Db2 = tf.Variable(tf.zeros(self.maxout_num * self.d_hidden2_nodes))
            self.DW3 = tf.Variable(tf.random_normal([self.d_hidden2_nodes, 1], stddev=0.01))
            self.Db3 = tf.Variable(tf.zeros(1))

    def generator(self, input):
        Gh1 = tf.nn.relu(tf.matmul(input, self.GW1) + self.Gb1)
        Gh2 = tf.nn.relu(tf.matmul(Gh1, self.GW2) + self.Gb2)
        return tf.nn.sigmoid(tf.matmul(Gh2, self.GW3) + self.Gb3)

    def discriminator(self, input):
        Dh1_in = tf.reshape(tf.matmul(input, self.DW1) + self.Db1, [-1, self.maxout_num, self.d_hidden1_nodes])
        Dh1_out = tf.nn.dropout(tf.reduce_max(Dh1_in, reduction_indices=[1]), self.keep_prob)  # maxout
        Dh2_in = tf.reshape(tf.matmul(Dh1_out, self.DW2) + self.Db2, [-1, self.maxout_num, self.d_hidden2_nodes])
        Dh2_out = tf.nn.dropout(tf.reduce_max(Dh2_in, reduction_indices=[1]), self.keep_prob)
        return tf.nn.sigmoid(tf.matmul(Dh2_out, self.DW3) + self.Db3)

    def train(self):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        G = self.generator(self.Z)
        DG = self.discriminator(G)
        Dloss = - tf.reduce_mean(tf.log(self.discriminator(self.X)) + tf.log(1 - DG))
        Gloss = -tf.reduce_mean(tf.log(DG + 1e-8))

        vars = tf.trainable_variables()
        Dvars = [v for v in vars if v.name.startswith("D")]
        Gvars = [v for v in vars if v.name.startswith("G")]

        Doptimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(Dloss, var_list=Dvars)
        Goptimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(Gloss, var_list=Gvars)

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        dloss = gloss = 0.0
        for i in range(self.iterations):
            start = time.clock()
            for j in range(self.batch_num):
                # update discriminator  k = 1
                x, _ = mnist.train.next_batch(self.batch_size)  # x_data
                z = np.random.uniform(-1, 1, size=(self.batch_size, self.noise_dim))  # noise
                loss, _ = sess.run([Dloss, Doptimizer], feed_dict={self.X: x, self.Z: z, self.keep_prob: 0.5})
                dloss += loss
                # update generator
                z = np.random.uniform(-1, 1, size=(self.batch_size, self.noise_dim))
                loss, _ = sess.run([Gloss, Goptimizer], feed_dict={self.Z: z, self.keep_prob: 1.0})
                gloss += loss
            end = time.clock()
            print("%d: dloss=%.5f, gloss=%.5f" % (i, dloss / self.batch_num, gloss / self.batch_num))
            print("time: ", end - start)
            if math.isnan(dloss) or math.isnan(gloss):
                sess.run(tf.initialize_all_variables())  # initialize & retry if NaN
            dloss = gloss = 0.0

        self.plot("GAN_MNIST.png", G, sess)

    def plot(self, path, G, sess):
        z = np.random.uniform(-1, 1, size=(self.generative_dim[0] * self.generative_dim[1], self.noise_dim))
        Gz = sess.run(G, feed_dict={self.Z: z})  # input noise and generate samples
        fig = plt.figure()
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        total = self.generative_dim[0] * self.generative_dim[1]
        for i in range(total):
            ax = fig.add_subplot(self.generative_dim[0], self.generative_dim[1], i + 1)
            ax.axis("off")
            ax.imshow(Gz[i, :].reshape((28, 28)), cmap=plt.get_cmap("gray"))
        fig.savefig(path)
        plt.close(fig)


def __main__():
    gan = GAN_MNIST()
    gan.train()

if __name__ == '__main__':
    __main__()

