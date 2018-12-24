import tensorflow as tf
import numpy as np

M = 2
K = 1
H = 2

Epoch_num = 1000


X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0],[1],[1],[0]])

x = tf.placeholder(tf.float32, shape=[None, M])
t = tf.placeholder(tf.float32, shape=[None, K])

W = tf.Variable(tf.random.truncated_normal([M, H]))
b = tf.Variable(tf.zeros([H]))
V = tf.Variable(tf.random.truncated_normal([H, K]))
c = tf.Variable(tf.zeros([K]))

h = tf.nn.sigmoid(tf.matmul(x, W) + b)
y = tf.nn.sigmoid(tf.matmul(h, V) + c)

cross_entropy = - tf.reduce_sum(t * tf.log(y) + (1 - t) * tf.log(1 - y))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(Epoch_num):
    sess.run(train_step, feed_dict={
        x: X,
        t: Y
    })

sess.close()


