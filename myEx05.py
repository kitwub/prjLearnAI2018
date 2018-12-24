import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import numpy as np
from sklearn.utils import shuffle

np.random.seed(0)
tf.set_random_seed(0)

M = 2       # input
K = 3       # class
n = 100     # train data(1 class)
N = n * K   # all data

minibatch_size = 50

X1 = np.random.randn(n, M) + np.array([0, 10])
X2 = np.random.randn(n, M) + np.array([5, 5])
X3 = np.random.randn(n, M) + np.array([10, 0])
X = np.concatenate((X1, X2, X3), axis=0)

Y1 = np.array([[1, 0, 0] for _ in range(n)])
Y2 = np.array([[0, 1, 0] for _ in range(n)])
Y3 = np.array([[0, 0, 1] for _ in range(n)])
Y = np.concatenate((Y1, Y2, Y3), axis=0)

model = Sequential()
model.add(Dense(input_dim=M, units=K))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1))

model.fit(X, Y, epochs=20, batch_size=minibatch_size)

X_s, Y_s = shuffle(X, Y)
# classes = model.predict_classes(X_s[0:10], batch_size=minibatch_size)
# prob = model.predict_proba(X_s[0:10], batch_size=minibatch_size)
classes = model.predict_classes(X_s[0:10])
prob = model.predict_proba(X_s[0:10])

print('classified:')
print(np.argmax(model.predict(X_s[0:10]), axis=1) == classes)
print()
print('output probability:')
print(prob)
pass
# W = tf.Variable(tf.zeros([M, K]))
# b = tf.Variable(tf.zeros([K]))
#
# x = tf.placeholder(tf.float32, shape=[None, M])
# t = tf.placeholder(tf.float32, shape=[None, K])
# y = tf.nn.softmax(tf.matmul(x, W) + b)
#
# # cross_entropy = tf.reduce_mean(
# #     - tf.reduce_sum(t * tf.log(y), axis=1)
# # )
# cross_entropy = tf.reduce_sum(
#     - tf.reduce_sum(t * tf.log(y), axis=1)
# )
#
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
#
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
#
# batch_size = 50
# n_batches = N
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# for epoch in range(20):
#     X_s, Y_s = shuffle(X, Y)
#
#     for i in range(n_batches):
#         start = i * batch_size
#         end = start + batch_size
#
#         sess.run(train_step, feed_dict={
#             x: X_s[start:end],
#             t: Y_s[start:end]
#         })
#
# X_s, Y_s = shuffle(X, Y)
#
# classified = correct_prediction.eval(session=sess, feed_dict={
#     x: X_s[0: 10],
#     t: Y_s[0: 10]
# })
#
# prob = y.eval(session=sess, feed_dict={
#     x: X_s[0: 10]
# })
#
# print('classified:')
# print(classified)
# print()
# print('probability')
# print(prob)
#
# sess.close()
# pass
