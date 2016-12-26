#authors : Guanchen He(ghe10), Yingyi Zhang(yingyiz2), Guxin Jin(gjin7)

import numpy as np
import tensorflow as tf

def readData(path):
    x = []
    y = []
    file = open(path)
    i = 0
    for lines in file:
        words = lines.split()
        num = int(words[0])
        tmp_y = []

        tmp_y = []
        if num == 0:
            tmp_y.append(1)
            tmp_y.append(0)
        else:
            tmp_y.append(0)
            tmp_y.append(1)

        f = open("hw1/" + words[1])
        raw_data = f.read()
        tmp_x = []
        tmp_x.append(1)
        valid = 1
        for number in raw_data.split():
            if np.isnan(float(number)):
                #print(float(number))
                valid = 0
                break
            tmp_x.append(float(number))
        f.close()
        if valid == 1:
            x.append(np.array(tmp_x))
            i += 1
            #tmp_y.append(num)
            y.append(tmp_y)
            #print(i)
    print("read finished")
    x = np.array(x, float)
    y = np.array(y, float)
    return x, y


x, y = readData("hw1/train/lab/hw1train_labels.txt")
X_valid, Y_valid = readData("hw1/dev/lab/hw1dev_labels.txt")
X_test, Y_test = readData("hw1/eval/lab/hw1eval_labels.txt")
print(type(x))
print(x.shape)
print(type(y))
print(y.shape)

f1 = open('results/tf_train.txt', 'w')
f2 = open('results/tf_valid.txt', 'w')
f3 = open('results/tf_test.txt', 'w')

X_placeholder = tf.placeholder(tf.float32, shape=[None,17]) # Define a placeholder for the inpu
y_placeholder = tf.placeholder(tf.float32, shape=[None,None]) # Define a placeholder for the labe
w = tf.Variable(tf.random_normal([17, 2])) # Random initialize the weight and bias
y_hat = tf.nn.softmax(tf.matmul(X_placeholder, w))

#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_placeholder * tf.log(y_hat), reduction_indices = [1]))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_placeholder * tf.log(y_hat), reduction_indices = [1]))
opt = tf.train.GradientDescentOptimizer(0.1)# Create a gradient descent optimizer
update_op = opt.minimize(cross_entropy)
sess = tf.Session() # Defines a session
sess.run(tf.initialize_all_variables())

correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y_placeholder, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Performs gradient descent
for step in range(0,6500):
    feed_dict = {X_placeholder:x, y_placeholder:y}
    loss_np = sess.run([update_op, cross_entropy], feed_dict=feed_dict)
    if step % 1 == 0:
        #print(sess.run(w))
        w_np = sess.run(w)
        f1.write(str(sess.run(accuracy, feed_dict={X_placeholder:x, y_placeholder:y})))
        f1.write("\n")

        f2.write(str(sess.run(accuracy, feed_dict = {X_placeholder:X_valid, y_placeholder:Y_valid})))
        f2.write("\n")

        f3.write(str(sess.run(accuracy, feed_dict={X_placeholder: X_test, y_placeholder: Y_test})))
        f3.write("\n")

    if step % 100 == 0:
        print(step)
        print("train accuracy")
        print(sess.run(accuracy, feed_dict={X_placeholder:x, y_placeholder:y}))
        ''''
        print("validate accuracy : ")
        print(sess.run(accuracy, feed_dict = {X_placeholder:X_valid, y_placeholder:Y_valid}))
        print("test accuracy : ")
        print(sess.run(accuracy, feed_dict={X_placeholder: X_test, y_placeholder: Y_test}))
        '''
f1.close()
f2.close()
f3.close()