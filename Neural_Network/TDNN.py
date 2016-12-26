import tensorflow as tf
import numpy as np
from read import readData

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def get_batch(X, Y, i, batch_size):
    bias = i * batch_size
    batch_x = []
    batch_y = []
    for j in range(batch_size):
        batch_x.append(X[bias + j])
        batch_y.append(Y[bias + j])
    x = np.array(batch_x).reshape(batch_size, 1120)
    y = np.array(batch_y).reshape(batch_size, 9)
    return x, y


x = tf.placeholder(tf.float32, shape=[None, 1120])
y = tf.placeholder(tf.float32, shape=[None, 9])

''' we have 70 frames, each time, we use conv to compute 3 frame, that is 3 * 16
    it seems that we need further down sampling????
'''
x_2d = tf.reshape(x, [-1, 70, 16, 1])

w_conv1 = weight_variable([3, 16, 1, 1]) # kernel size: 3 * 16,  input feature number (similar to last layer kernel number): 1, kernel number : 32
b_conv1 = bias_variable([1])

y_conv1 = tf.nn.sigmoid(conv2d(x_2d, w_conv1) + b_conv1)
y_pool1 = max_pool_2x2(y_conv1) # now we should have y : 35 * 8

w_conv2 = weight_variable([5, 8, 1, 1])
b_conv2 = bias_variable([1])

y_conv2 = tf.nn.sigmoid(conv2d(y_pool1, w_conv2) + b_conv2)
y_pool2 = max_pool_2x2(y_conv2)

w_conv3 = weight_variable([2, 3, 1, 1])
b_conv3 = bias_variable([1])

y_conv3 = tf.nn.sigmoid(conv2d(y_pool2, w_conv3) + b_conv3)
#y_pool3 = max_pool_2x2(y_conv3)

w_fc = weight_variable([72 * 1, 9])
b_fc = weight_variable([9])

y_pool3_flat = tf.reshape(y_conv3, [-1, 72 * 1])  # without pooling3 72
y_hat = tf.nn.softmax(tf.matmul(y_pool3_flat, w_fc) + b_fc)

cross_entropy = tf.reduce_mean(- tf.reduce_sum(y * tf.log(y_hat), reduction_indices = [1]))
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

epoch = 100
batch_size = 200
X, Y, length = readData("hw1/train/lab/hw2train_labels.txt")
batches = (int)(length / batch_size)

for i in range(epoch):
    for j in range(batches):
        #i_bias = i * batch_size
        x_batch, y_batch = get_batch(X, Y, j, batch_size)
        #train_step.run(feed_dict={x: x_batch, y: y_batch})
        feed_dict = {x: x_batch, y: y_batch}
        loss_np = sess.run([train_step, cross_entropy], feed_dict=feed_dict)
    if i % 10 == 0:

        #train_accuracy = accuracy.eval(feed_dict={x: X, y: Y})
        train_x, train_y = get_batch(X, Y, 0, length)
        train_accuracy = sess.run(accuracy, feed_dict={x: train_x , y: train_y})
        print(train_accuracy)