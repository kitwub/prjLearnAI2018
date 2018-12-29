from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf


np.random.seed(0)
tf.set_random_seed(1234)

N = 300
M = 2
K = 1
H1 = 3

Epoch_num = 1000
minibatch_size = 20

X, y = datasets.make_moons(N, noise=0.3)
# print(X, y)
Y = y.reshape(N, 1)

X_train, X_test, Y_train, Y_test = \
    train_test_split(X, Y, train_size=0.8)


W = tf.Variable(tf.random.truncated_normal([M, H1]))
b = tf.Variable(tf.zeros([H1]))

V = tf.Variable(tf.random.truncated_normal([H1, K]))
c = tf.Variable(tf.zeros([K]))


x = tf.placeholder(dtype=tf.float32, shape=[None, M])
t = tf.placeholder(dtype=tf.float32, shape=[None, K])

h = tf.nn.sigmoid(tf.matmul(x, W) + b)
y = tf.nn.sigmoid(tf.matmul(h, V) + c)
# y = tf.reduce_mean(
#     tf.nn.sigmoid(tf.matmul(h, V) + c), reduction_indices=0
# )

# cross_entropy = - tf.reduce_mean(t * tf.log(y) + (1 - t) * tf.log(1 - y), reduction_indices=0)
cross_entropy = - tf.reduce_sum(t * tf.log(y) + (1 - t) * tf.log(1 - y))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)
accuracy = tf.reduce_mean(tf.to_float(correct_prediction))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(Epoch_num):
    X_s, Y_s = shuffle(X_train, Y_train)

    for i in range(len(Y_train) // minibatch_size):
        start = minibatch_size * i
        end = start + minibatch_size

        sess.run(train_step, feed_dict={
            x: X_s[start:end],
            t: Y_s[start:end]
        })

accuracy_rate = accuracy.eval(session=sess, feed_dict={
    x: X_test,
    t: Y_test
})
print('accuracy:', accuracy_rate)

sess.close()