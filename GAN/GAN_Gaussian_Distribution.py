# Yingyi Zhang  (yingyiz2) Date: 2016-12-1

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class GAN():
    def __init__(self):
        self.learning_rate = 0.005
        self.batch_size = 100
        self.iterations = 3000
        self.mean = 0
        self.std = 1
        self.hidden1_nodes = 10
        self.hidden2_nodes = 10

    def generator_net(self, input, output_dim):
        w1 = tf.get_variable("gw0", [input.get_shape()[1], self.hidden1_nodes],
                             initializer=tf.random_normal_initializer())  # hidden node = 6
        b1 = tf.get_variable("gb0", [self.hidden1_nodes], initializer=tf.constant_initializer(0.0))
        w2 = tf.get_variable("gw1", [self.hidden1_nodes, self.hidden2_nodes], initializer=tf.random_normal_initializer())
        b2 = tf.get_variable("gb1", [self.hidden2_nodes], initializer=tf.constant_initializer(0.0))  # second layer hidden node = 5
        w3 = tf.get_variable("gw2", [self.hidden2_nodes, output_dim], initializer=tf.random_normal_initializer())
        b3 = tf.get_variable("gb2", [output_dim], initializer=tf.constant_initializer(0.0))

        layer1 = tf.nn.tanh(tf.matmul(input, w1) + b1)  # tanh as activation function
        layer2 = tf.nn.tanh(tf.matmul(layer1, w2) + b2)
        g_out = tf.nn.tanh(tf.matmul(layer2, w3) + b3)
        return g_out, [w1, b1, w2, b2, w3, b3]

    def discriminator_net(self, input, output_dim):
        w1 = tf.get_variable("dw0", [input.get_shape()[1], self.hidden1_nodes],
                             initializer=tf.random_normal_initializer())
        b1 = tf.get_variable("db0", [self.hidden1_nodes], initializer=tf.constant_initializer(0.0))
        w2 = tf.get_variable("dw1", [self.hidden1_nodes, self.hidden2_nodes], initializer=tf.random_normal_initializer())
        b2 = tf.get_variable("db1", [self.hidden2_nodes], initializer=tf.constant_initializer(0.0))  # second layer hidden node = 5
        w3 = tf.get_variable("dw2", [self.hidden2_nodes, output_dim], initializer=tf.random_normal_initializer())
        b3 = tf.get_variable("db2", [output_dim], initializer=tf.constant_initializer(0.0))

        layer1 = tf.nn.tanh(tf.matmul(input, w1) + b1)  # tanh as activation function
        layer2 = tf.nn.tanh(tf.matmul(layer1, w2) + b2)
        d_out = tf.nn.tanh(tf.matmul(layer2, w3) + b3)
        return d_out, [w1, b1, w2, b2, w3, b3]

    def optimizer_func(self, loss, var_list):
        batch = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
            0.001,                     # Base learning rate.
            batch,                     # Current index of the dataset.
            int(self.iterations / 4),  # Decay step - this decays 4 times throughout training process.
            0.95,  # Decay rate.
            staircase=True)
        opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=batch,var_list=var_list)
        # opt = tf.train.MomentumOptimizer(learning_rate, 0.5).minimize(loss, global_step=batch, var_list=var_list)
        return opt

    def pre_train(self):
        with tf.variable_scope("D_pre"):
            input_node = tf.placeholder(tf.float32, shape=(self.batch_size, 1))  # M -- batch size
            train_labels = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            D, theta = self.discriminator_net(input_node, 1)  # theta is trainable variables such as W, b
            loss = tf.reduce_mean(tf.square(D - train_labels))
        optimizer = self.optimizer_func(loss, None)
        sess = tf.InteractiveSession()
        tf.initialize_all_variables().run()
        # lh = np.zeros(1000)
        for i in range(1000):
            # instead of sampling only from gaussian, want the domain to be covered as uniformly as possible
            d = (np.random.random(self.batch_size) - 0.5) * 10.0  # generate M = 200 random number in range of [-5, 5]
            labels = norm.pdf(d, loc=self.mean, scale=self.std)  # generate normal data set as input
            sess.run(
                [loss, optimizer],
                {input_node: np.reshape(d, (self.batch_size, 1)), train_labels: np.reshape(labels, (self.batch_size, 1))}
            )
        weightsD = sess.run(theta)  # copy the learned weights over into a tmp array
        sess.close()
        return weightsD

    def train(self):
        with tf.variable_scope("G"):
            z_node = tf.placeholder(tf.float32, shape=(self.batch_size, 1))  # M uniform 01 floats
            G, theta_g = self.generator_net(z_node, 1)  # generate normal transformation of Z
            G = tf.mul(5.0, G)  # scale up by 5 to match range
        with tf.variable_scope("D") as scope:
            x_node = tf.placeholder(tf.float32, shape=(self.batch_size, 1))  # input M normally distributed floats
            d_out, theta_d = self.discriminator_net(x_node, 1)  # likelihood
            D1 = tf.maximum(tf.minimum(d_out, 0.99), 0.01)  # normalized as probability
            # make a copy of D that uses the same variables, but takes in G as input
            scope.reuse_variables()
            d_out, theta_d = self.discriminator_net(G, 1)
            D2 = tf.maximum(tf.minimum(d_out, 0.99), 0.01)
        obj_d = tf.reduce_mean(tf.log(D1) + tf.log(1 - D2))  # maximize
        obj_g = tf.reduce_mean(tf.log(D2))                   # maximize
        opt_d = self.optimizer_func(1 - obj_d, theta_d)  # theta_d is what need to be trained
        opt_g = self.optimizer_func(1 - obj_g, theta_g)  # maximize log(D(G(z)))
        sess = tf.InteractiveSession()
        tf.initialize_all_variables().run()
        weightsD = self.pre_train()
        # copy weights from pre-training over to new D network
        for i, v in enumerate(theta_d):
            sess.run(v.assign(weightsD[i]))
        k = 1
        histd, histg = np.zeros(self.iterations), np.zeros(self.iterations)
        for i in range(self.iterations):
            # update discriminator
            for j in range(k):
                x = np.random.normal(self.mean, self.std, self.batch_size)  # sampled m-batch from p_data
                x.sort()
                z = np.linspace(-5.0, 5.0, self.batch_size) + np.random.random(self.batch_size) * 0.01   # sample m-batch from noise prior
                histd[i], tmp = sess.run(
                    [obj_d, opt_d],
                    {x_node: np.reshape(x, (self.batch_size, 1)), z_node: np.reshape(z, (self.batch_size, 1))}
                )

            # update generator
            z = np.linspace(-5.0, 5.0, self.batch_size) + np.random.random(self.batch_size) * 0.01  # sample noise prior
            histg[i], tmp = sess.run([obj_g, opt_g], {z_node: np.reshape(z, (self.batch_size, 1))})  # update generator
            if i % (int(self.iterations / 10)) == 0:
                print(i)
        self.plot(D1, x_node, G, z_node, sess)
        sess.close()

    # plots pg, p_data, discriminator boundary
    def plot(self, D1, x_node, G, z_node, sess):
        f, ax = plt.subplots(1)
        # p_data
        xs = np.linspace(-5, 5, 1000)
        ax.plot(xs, norm.pdf(xs, loc=self.mean, scale=self.std), label='p_x')

        # decision boundary
        r = 5000  # resolution (number of points)
        xs = np.linspace(-5, 5, r)
        ds = np.zeros((r, 1))  # decision surface
        # process multiple points in parallel in same mini batch
        for i in range(int(r / self.batch_size)):
            x = np.reshape(xs[self.batch_size * i: self.batch_size * (i + 1)], (self.batch_size, 1))
            ds[self.batch_size * i: self.batch_size * (i + 1)] = sess.run(D1, {x_node: x})

        ax.plot(xs, ds, label='discriminator boundary')

        zs = np.linspace(-5, 5, r)
        gs = np.zeros((r, 1))  # generator function
        for i in range(int(r / self.batch_size)):
            z = np.reshape(zs[self.batch_size * i: self.batch_size * (i + 1)], (self.batch_size, 1))
            gs[self.batch_size * i: self.batch_size * (i + 1)] = sess.run(G, {z_node: z})
        histc, edges = np.histogram(gs, bins=10)
        ax.plot(np.linspace(-5, 5, 10), histc / float(r), label='p_g')

        ax.set_ylim(0, 1.1)
        plt.legend()
        plt.show()


def __main__():
    gan = GAN()
    gan.train()

if __name__ == '__main__':
    __main__()


