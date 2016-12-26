# Yingyi Zhang (yingyiz2)  Date: 2016-12-1
'''
The CIFAR data set is supposed to be placed with source file in
the same directory
'''

import cPickle as pickle
import matplotlib
matplotlib.use('Agg')
import numpy as np
import math, time
import scipy.io
import tensorflow as tf
import matplotlib.pyplot as plt

def load_CIFAR_batch(filename):
  with open(filename, 'rb') as f:
    datadict = pickle.load(f)
    X = datadict['data']
    Y = datadict['labels']
    print(np.shape(X))
    X = X.reshape(10000, 3, 32, 32) #.transpose(2, 3, 1, 0).astype("float")
    Y = np.array(Y)
    return X, Y


class DCGAN_CIFAR():
    def __init__(self, input):
        self.noise_dim = 100
        self.Ghidden = [512, 256, 128] # Generator hidden nodes 3 layers
        self.Dhidden = [64, 128, 256]  # Discriminator hidden nodes 3 layers
        # self.Ghidden = [256, 128, 64] # Generator hidden nodes 3 layers
        # self.Dhidden = [32, 64, 128]  # Discriminator hidden nodes 3 layers

        self.batch_size = 50
        self.generative_dim = (5, 6)  # samples drawing size
        self.generative_total = self.generative_dim[0] * self.generative_dim[1]
        self.epoch = 10

        self.train_data = input
        self.fig_width = 0  #32
        self.fig_height = 0 #32
        self.n_channels = 0 #3
        self.total = 0      #10000
        self.batch_num = 0 #self.total // self.batch_size
        self.pre_process()

        self.X = tf.placeholder(tf.float32, shape=(None, self.fig_width, self.fig_height, self.n_channels))
        self.Z = tf.placeholder(tf.float32, shape=(None, self.noise_dim))
        self.keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope("G"):
            self.GW0 = tf.Variable(tf.random_normal([self.noise_dim, self.Ghidden[0] * 4 * 4], stddev=0.01))
            self.Gb0 = tf.Variable(tf.zeros(self.Ghidden[0]))
            self.GW1 = tf.Variable(tf.random_normal([5, 5, self.Ghidden[1], self.Ghidden[0]], stddev=0.01))
            self.Gb1 = tf.Variable(tf.zeros(self.Ghidden[1]))
            self.GW2 = tf.Variable(tf.random_normal([5, 5, self.Ghidden[2], self.Ghidden[1]], stddev=0.01))
            self.Gb2 = tf.Variable(tf.zeros(self.Ghidden[2]))
            self.GW3 = tf.Variable(tf.random_normal([5, 5, self.n_channels, self.Ghidden[2]], stddev=0.01))
            self.Gb3 = tf.Variable(tf.zeros(self.n_channels))

        with tf.variable_scope("D"):
            self.DW0 = tf.Variable(tf.random_normal([5, 5, self.n_channels, self.Dhidden[0]], stddev=0.01))
            self.Db0 = tf.Variable(tf.zeros(self.Dhidden[0]))
            self.DW1 = tf.Variable(tf.random_normal([5, 5, self.Dhidden[0], self.Dhidden[1]], stddev=0.01))
            self.Db1 = tf.Variable(tf.zeros(self.Dhidden[1]))
            self.DW2 = tf.Variable(tf.random_normal([5, 5, self.Dhidden[1], self.Dhidden[2]], stddev=0.01))
            self.Db2 = tf.Variable(tf.zeros(self.Dhidden[2]))
            self.DW3 = tf.Variable(tf.random_normal([(self.fig_width // 8) * (self.fig_height // 8) * self.Dhidden[2], 1], stddev=0.01))
            self.Db3 = tf.Variable(tf.zeros(1))

    def pre_process(self):
        self.fig_width, self.fig_height, self.n_channels, self.total = self.train_data.shape
        self.train_data = self.train_data.reshape(self.fig_width * self.fig_height * self.n_channels, self.total)
        self.train_data -= self.train_data.min(axis=0)
        self.train_data = (np.array(self.train_data, dtype=np.float32) /
                           self.train_data.max(axis=0)).T.reshape(self.total, self.fig_width, self.fig_height, self.n_channels)
        self.batch_num = self.total // self.batch_size
        print(self.fig_width, self.fig_height, self.n_channels, self.total)

    # batch normalization & relu
    def batch_normalize_RELU(self, data):
        mean, variance = tf.nn.moments(data, axes=[0, 1, 2])
        return tf.nn.relu(tf.nn.batch_normalization(data, mean, variance, None, None, 1e-5))

    # batch normalization & leaky relu
    def batch_normalize_leakyRELU(self, data, a=0.2):
        mean, variance = tf.nn.moments(data, axes=[0, 1, 2])
        b = tf.nn.batch_normalization(data, mean, variance, None, None, 1e-5)
        return tf.maximum(a * b, b)

    def generator(self, input):
        # print(self.fig_height)
        Gh0 = self.batch_normalize_RELU(
            tf.nn.bias_add(
                tf.reshape(tf.matmul(input, self.GW0), [-1, self.fig_width // 8, self.fig_height // 8, self.Ghidden[0]]), self.Gb0
            )
        )
        Gh1 = self.batch_normalize_RELU(
            tf.nn.bias_add(
                tf.nn.conv2d_transpose(Gh0, self.GW1, [self.batch_size, self.fig_width // 4, self.fig_height // 4, self.Ghidden[1]], [1, 2, 2, 1]), self.Gb1
            )
        )
        # print(self.GW2.get_shape()[2])
        Gh2 = self.batch_normalize_RELU(
            tf.nn.bias_add(
                tf.nn.conv2d_transpose(Gh1, self.GW2, [self.batch_size, self.fig_width // 2, self.fig_height // 2, self.Ghidden[2]], [1, 2, 2, 1]), self.Gb2
            )
        )
        # print(self.n_channels)
        # print(self.GW3.get_shape()[2])
        G = tf.nn.tanh(
            tf.nn.bias_add(
                tf.nn.conv2d_transpose(Gh2, self.GW3, [self.batch_size, self.fig_width, self.fig_height, self.n_channels], [1, 2, 2, 1]), self.Gb3
            )
        )
        return G

    def discriminator(self, input):
        Dh0 = self.batch_normalize_leakyRELU(
            tf.nn.bias_add(tf.nn.conv2d(input, self.DW0, [1, 2, 2, 1], padding='SAME'), self.Db0)
        )
        Dh1 = self.batch_normalize_leakyRELU(
            tf.nn.bias_add(tf.nn.conv2d(Dh0, self.DW1, [1, 2, 2, 1], padding='SAME'), self.Db1)
        )
        Dh2 = self.batch_normalize_leakyRELU(
            tf.nn.bias_add(tf.nn.conv2d(Dh1, self.DW2, [1, 2, 2, 1], padding='SAME'), self.Db2)
        )
        return tf.nn.sigmoid(
            tf.matmul(tf.reshape(Dh2, [-1, (self.fig_width // 8) * (self.fig_height // 8) * self.Dhidden[2]]), self.DW3) + self.Db3)

    def train(self):
        G = self.generator(self.Z)
        DG = self.discriminator(G)
        Dloss = -tf.reduce_mean(tf.log(self.discriminator(self.X)) + tf.log(1 - DG))
        Gloss = tf.reduce_mean(tf.log(1 - DG) - tf.log(DG + 1e-9))  # the second term for stable learning

        vars = tf.trainable_variables()
        Dvars = [v for v in vars if v.name.startswith("D")]
        Gvars = [v for v in vars if v.name.startswith("G")]
        print(self.total)
        Doptimizer = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(Dloss, var_list=Dvars)
        Goptimizer = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(Gloss, var_list=Gvars)
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        generative_z = np.random.uniform(-1, 1, size=(self.batch_size, self.noise_dim))  # nsamples < mini_batch_size
        for e in range(self.epoch):
            t0 = time.time()
            index = np.random.permutation(self.total)
            dloss = gloss = 0.0
            for i in range(self.batch_num):
                x = self.train_data[index[i * self.batch_size:(i + 1) * self.batch_size], :]
                z = np.random.uniform(-1, 1, size=(self.batch_size, self.noise_dim))
                loss, _ = sess.run([Dloss, Doptimizer], feed_dict={self.X: x, self.Z: z, self.keep_prob: 0.5})
                dloss += loss
                z = np.random.uniform(-1, 1, size=(self.batch_size, self.noise_dim))
                loss, _ = sess.run([Gloss, Goptimizer], feed_dict={self.Z: z, self.keep_prob: 1.0})
                gloss += loss

                if math.isnan(dloss) or math.isnan(gloss):
                    sess.run(tf.initialize_all_variables())  # initialize & retry if NaN
                    dloss = gloss = 0.0
            print("%d: dloss=%.5f, gloss=%.5f, time=%.1f" % (e + 1, dloss / self.batch_num, gloss / self.batch_num, time.time() - t0))
            self.plot("png/dcgan-cifar-%03d.png" % (e + 1), G, generative_z, sess)

    def plot(self, filepath, G, z, sess):
        Gz = sess.run(G, feed_dict={self.Z: z})
        fig = plt.gcf()
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        for i in range(self.generative_total):
            ax = fig.add_subplot(self.generative_dim[0], self.generative_dim[1], i + 1)
            ax.axis("off")
            ax.imshow(Gz[i, :, :, :])
        plt.savefig(filepath)
        plt.draw()
        plt.pause(0.01)

def __main__():
    train_data, label = load_CIFAR_batch("cifar-10-batches-py/data_batch_1")
    train_data2, label = load_CIFAR_batch("cifar-10-batches-py/data_batch_2")
    train_data3, label = load_CIFAR_batch("cifar-10-batches-py/data_batch_3")
    train_data4, label = load_CIFAR_batch("cifar-10-batches-py/data_batch_4")
    train_data5, label = load_CIFAR_batch("cifar-10-batches-py/data_batch_5")
    train_data = np.concatenate((train_data, train_data2), axis=0)
    train_data = np.concatenate((train_data, train_data3), axis=0)
    train_data = np.concatenate((train_data, train_data4), axis=0)
    train_data = np.concatenate((train_data, train_data5), axis=0)
    train_data = train_data.transpose(2, 3, 1, 0).astype("float")
    gan = DCGAN_CIFAR(train_data)
    # gan.pre_process()
    gan.train()

if __name__ == '__main__':
    __main__()

