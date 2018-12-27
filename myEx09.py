from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Activation
# import numpy as np
# from sklearn.utils import shuffle
from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="0", # specify GPU number
        allow_growth=True
    )
)
set_session(tf.Session(config=config))

N = 1000
data_noise = 0.3

M = 2
K = 1
H1 = 3
Epoch_num = 40
minibatch_size = 20



X, y = datasets.make_moons(N, noise=data_noise)
# Y = y.reshape(N, 1)

X_train, X_test, Y_train, Y_test = \
    train_test_split(X, y, train_size=0.8)

model = Sequential()

model.add(Dense(H1, input_dim=M))
model.add(Activation('sigmoid'))

model.add(Dense(K))
model.add(Activation('sigmoid'))

model.compile(
    SGD(lr=0.1),
    loss='binary_crossentropy',
    metrics=['accuracy']
    )

model.fit(X_train, Y_train, batch_size=minibatch_size, epochs=Epoch_num)

loss_and_metrics = model.evaluate(X_test, Y_test)
print(loss_and_metrics)

pass
# pred = model.predict_classes(X_test)
# prob = model.predict_proba(X_test)
#
# print('pred:', pred)
# print('prob:', prob)