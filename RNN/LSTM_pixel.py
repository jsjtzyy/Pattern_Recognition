# This code is completed by ghe10, yingyiz2 and gjin7

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell

def read_Data():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

    # downsample by max pooling method
    input = tf.reshape(trX, [-1, 28, 28, 1])
    op = tf.nn.max_pool(input, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')
    testinput = tf.reshape(teX, [-1, 28, 28, 1])
    testop = tf.nn.max_pool(testinput, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')
    res = tf.reshape(op, [-1, 49])
    test_res = tf.reshape(testop, [-1, 49])
    with tf.Session() as sess:
        trX = sess.run(res)
        teX = sess.run(test_res)
    print("Pooling finished")
    return trX, trY, teX, teY

class RNN():
    def __init__(self):
        self.learning_rate = 0.01
        self.batch_size = 100
        self.T = 49
        self.input_size = 1
        self.hidden_nodes = 100

        # weight matrix
        self.weights = {
             'W_in' : tf.Variable(tf.random_normal([self.input_size, self.hidden_nodes])),
             'W_out' : tf.Variable(tf.random_normal([self.hidden_nodes, 10]))}
        # offset vector
        self.bs = {
            'b_in' : tf.Variable(tf.zeros([self.hidden_nodes, ])),
            'b_out' : tf.Variable(tf.zeros([10, ]))
        }

        self.x = tf.placeholder(tf.float32, [None, self.T, self.input_size])
        X = tf.reshape(self.x, [-1, self.input_size])
        x_hidden = tf.matmul(X, self.weights['W_in']) + self.bs['b_in']
        self.hidden_value = tf.reshape(x_hidden, [-1, self.T, self.hidden_nodes])
        self.y = tf.placeholder(tf.float32, [None, 10]) # define placeholder for labels

        # define single cell in LSTM network
        self.time_delay_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_nodes, forget_bias=1.0, state_is_tuple=True)
        init_state = self.time_delay_cell.zero_state(self.batch_size, dtype=tf.float32) # initial state
        # update the cell based on previous state
        self.time_delay_cell_output, state =\
            tf.nn.dynamic_rnn(self.time_delay_cell, self.hidden_value, initial_state= init_state, time_major=False)

        self.unpacked_out = tf.unpack(tf.transpose(self.time_delay_cell_output, [1, 0, 2]))
        self.pred = tf.matmul(self.unpacked_out[-1], self.weights['W_out']) + self.bs['b_out']  # forward prediction
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.pred, self.y))
        opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)  # Create a gradient descent optimizer
        self.update_op = opt.minimize(loss)

        correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.sess = tf.Session()  # Defines a session
        self.sess.run(tf.initialize_all_variables())

    def train(self, epochs):
        accuracy_result = []  # store the accuracy at each iteration
        for epoch in range(epochs):
            train_accuracy = 0
            for index in range(int(55000 / self.batch_size)):
                batch_x = trX[index * self.batch_size: (index + 1) * self.batch_size]
                batch_x = batch_x.reshape([-1, self.T, self.input_size])
                batch_y = trY[index * self.batch_size: (index + 1) * self.batch_size]
                self.sess.run(self.update_op, feed_dict={self.x : batch_x, self.y : batch_y})
                '''
                if index % 1 == 0:
                    reshape_train_x = trX.reshape([-1, self.T, self.input_size])
                    print("train accuracy")
                    print(self.sess.run(self.accuracy, feed_dict={self.x: reshape_train_x, self.y : trY}))
                    reshape_test_x = teX.reshape([-1, self.T, self.input_size])
                    print("test accuracy")
                    print(self.sess.run(self.accuracy, feed_dict={self.x: reshape_test_x, self.y: trY}))
                '''
            # compute the accuracy
            print(epoch)
            for index in range(int(55000 / self.batch_size)):
                batch_x = trX[index * self.batch_size: (index + 1) * self.batch_size]
                batch_x = batch_x.reshape([-1, self.T, self.input_size])
                batch_y = trY[index * self.batch_size: (index + 1) * self.batch_size]
                train_accuracy += self.sess.run(self.accuracy, feed_dict={self.x: batch_x, self.y: batch_y})
            train_accuracy /= int(55000 / self.batch_size)
            accuracy_result.append(train_accuracy)
            print(train_accuracy)

        # evaluate the model on test dataset
        test_accuracy = 0
        for index in range(int(5000 / self.batch_size)):
            batch_x = teX[index * self.batch_size: (index + 1) * self.batch_size]
            batch_x = batch_x.reshape([-1, self.T, self.input_size])
            batch_y = teY[index * self.batch_size: (index + 1) * self.batch_size]
            test_accuracy += self.sess.run(self.accuracy, feed_dict={self.x: batch_x, self.y: batch_y})
        print("test accuracy")
        print(test_accuracy / (5000 / self.batch_size))

        np.savetxt("LSTM_pixel.txt", accuracy_result)


trX, trY, teX, teY = read_Data()
rnn = RNN()
rnn.train(20)






