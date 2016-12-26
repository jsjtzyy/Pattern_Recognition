'''This tensorflow part is completed by yingyiz2, ghe10 and gjin7'''
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from plot import plot_confusion_matrix_train, plot_confusion_matrix_eval
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
    mnist.test.labels

batch_size = 100

def train():
    x_placeholder = tf.placeholder(tf.float32, shape=[None, 784])
    y_placeholder = tf.placeholder(tf.float32, shape=[None, 10])  # Define a placeholder for the label

    w = tf.Variable(tf.random_normal([784, 10]))  # Random initialize the weight and bias
    b = tf.Variable(tf.zeros([10]))
    y_hat = tf.nn.softmax(tf.matmul(x_placeholder, w) + b)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_placeholder * tf.log(y_hat), reduction_indices=[1]))
    opt = tf.train.GradientDescentOptimizer(0.1)  # Create a gradient descent optimizer
    update_op = opt.minimize(cross_entropy)
    sess = tf.Session()  # Defines a session
    sess.run(tf.initialize_all_variables())

    correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y_placeholder, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    for step in range(0, 100):
        for index in range(55000 / batch_size - 1):
            batch_x = trX[index: index + batch_size]
            batch_y = trY[index: index + batch_size]
            feed_dict = {x_placeholder: batch_x, y_placeholder: batch_y}
            loss_np = sess.run([update_op, cross_entropy], feed_dict=feed_dict)
        if step % 1 == 0:
            print(step)
            print("train accuracy")
            print(sess.run(accuracy, feed_dict={x_placeholder: trX, y_placeholder: trY}))
            print("test  accuracy")
            print(sess.run(accuracy, feed_dict={x_placeholder: teX, y_placeholder: teY}))

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



if __name__ == "__main__":
    train()