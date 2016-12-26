'''This tensorflow part is completed by yingyiz2, ghe10 and gjin7'''
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from plot import plot_confusion_matrix_train, plot_confusion_matrix_eval
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
'''rbm with one hidden layer'''

def sample_prob(probability):
    return tf.nn.relu(tf.sign(probability - tf.random_uniform(tf.shape(probability))))

learning_rate = 1.0
batchsize = 100

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
    mnist.test.labels


class rbm_layer():
    '''init rbm layer'''
    def __init__(self, input_size = 784, hidden_nodes = 200, x = None):
        self.hidden_nodes = hidden_nodes
        if x == None:
            self.X = tf.placeholder("float", [None, input_size])
            print "x is none"
        else:
            self.X = x
        '''parameters'''
        self.w = tf.placeholder("float", [input_size, hidden_nodes])
        self.vb = tf.placeholder("float", [input_size])
        self.hb = tf.placeholder("float", [hidden_nodes])
        '''tf graph for unsupervised learning'''
        h0 = sample_prob(tf.nn.sigmoid(tf.matmul(self.X, self.w) + self.hb))
        v1 = sample_prob(tf.nn.sigmoid(
            tf.matmul(h0, tf.transpose(self.w)) + self.vb))
        h1 = tf.nn.sigmoid(tf.matmul(v1, self.w) + self.hb)
        w_positive_grad = tf.matmul(tf.transpose(self.X), h0)
        w_negative_grad = tf.matmul(tf.transpose(v1), h1)
        self.update_w = self.w + learning_rate * \
             (w_positive_grad - w_negative_grad) / tf.to_float(tf.shape(self.X)[0])
        self.update_vb = self.vb + learning_rate * tf.reduce_mean(self.X - v1, 0)
        self.update_hb = self.hb + learning_rate * tf.reduce_mean(h0 - h1, 0)

        h_sample = sample_prob(tf.nn.sigmoid(tf.matmul(self.X, self.w) + self.hb))
        v_sample = sample_prob(tf.nn.sigmoid(tf.matmul(h_sample, tf.transpose(self.w)) + self.vb))
        err = self.X - v_sample
        self.err_sum = tf.reduce_mean(err * err)
        '''the value used for those parameters'''
        self.new_w = np.zeros([input_size, hidden_nodes], np.float32)
        self.new_vb = np.zeros([input_size], np.float32)
        self.new_hb = np.zeros([hidden_nodes], np.float32)
        self.old_w = np.zeros([input_size, hidden_nodes], np.float32)
        self.old_vb = np.zeros([input_size], np.float32)
        self.old_hb = np.zeros([hidden_nodes], np.float32)

        self.sess = None

    def unsupervised_train(self):
        '''unsupervised learning process'''
        self.sess = tf.Session()
        init = tf.initialize_all_variables()
        self.sess.run(init)
        print self.sess.run(
            self.err_sum, feed_dict={self.X: trX, self.w: self.old_w, self.vb: self.old_vb, self.hb: self.old_hb})

        for start, end in zip(
            range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
            batch = trX[start:end]
            '''training and updating'''
            self.new_w = self.sess.run(self.update_w, feed_dict={
                self.X: batch, self.w: self.old_w, self.vb: self.old_vb, self.hb: self.old_hb})
            self.new_vb = self.sess.run(self.update_vb, feed_dict={
                self.X: batch, self.w: self.old_w, self.vb: self.old_vb, self.hb: self.old_hb})
            self.new_hb = self.sess.run(self.update_hb, feed_dict={
                self.X: batch, self.w: self.old_w, self.vb: self.old_vb, self.hb: self.old_hb})
            self.old_w = self.new_w
            self.old_vb = self.new_vb
            self.old_hb = self.new_hb
            '''print result'''
            if start % 10000 == 0:
                print self.sess.run(
                    self.err_sum, feed_dict={self.X: trX, self.w: self.new_w, self.vb: self.new_vb, self.hb: self.new_hb})

    def train(self, batch_size = 100):
        '''supervised learning based on the result of rbm'''
        x_placeholder = tf.placeholder(tf.float32, shape=[None, 784])
        y_placeholder = tf.placeholder(tf.float32, shape=[None, 10])
        w1 = tf.Variable(self.old_w)
        b1 = tf.Variable(self.old_hb)
        y1 = tf.nn.sigmoid(tf.matmul(x_placeholder, w1) + b1)

        w = tf.Variable(tf.random_normal([self.hidden_nodes, 10]))
        b = tf.Variable(tf.zeros([10]))
        y_hat = tf.nn.softmax(tf.matmul(y1, w) + b)

        '''define error function and optimization'''
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_placeholder * tf.log(y_hat), reduction_indices=[1]))
        opt = tf.train.GradientDescentOptimizer(0.1)
        update_op = opt.minimize(cross_entropy)
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y_placeholder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        for step in range(0, 30):
            for index in range(55000 / batch_size - 1):
                batch_x = trX[index : index + batch_size]
                batch_y = trY[index : index + batch_size]
                feed_dict = {x_placeholder: batch_x, y_placeholder: batch_y}
                loss = sess.run([update_op, cross_entropy], feed_dict=feed_dict)
            if step % 1 == 0:
                print(step)
                print("train accuracy")
                print(sess.run(accuracy, feed_dict={x_placeholder: trX, y_placeholder: trY}))
                print("test  accuracy")
                print(sess.run(accuracy, feed_dict={x_placeholder: teX, y_placeholder: teY}))
        '''the following code is for confusion matrix'''
        pred_trY = sess.run(y_hat, feed_dict={x_placeholder: trX, y_placeholder: trY})
        pred_teY = sess.run(y_hat, feed_dict={x_placeholder: teX, y_placeholder: teY})
        pred_trY_1d = np.argmax(pred_trY, 1)
        trY_1d = np.argmax(trY, 1)
        pred_teY_1d = np.argmax(pred_teY, 1)
        teY_1d = np.argmax(teY, 1)
        print "plot"
        index = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        plt.figure()
        cm = confusion_matrix((trY_1d), (pred_trY_1d))
        plot_confusion_matrix_train(cm, index)
        plt.show()

        plt.figure()
        cm = confusion_matrix((teY_1d), (pred_teY_1d))
        plot_confusion_matrix_eval(cm, index)
        plt.show()


if __name__=="__main__":
    rbm = rbm_layer()
    for i in range(2):
         rbm.unsupervised_train()
    for i in range(1):
        rbm.train()
