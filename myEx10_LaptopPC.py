from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.optimizers import SGD
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

set_session(
    tf.Session(
        config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                visible_device_list="2",  # (bad effect?)specify GPU number
                allow_growth=True
            ))))
# config = tf.ConfigProto(
#     gpu_options=tf.GPUOptions(
#         visible_device_list="0", # specify GPU number
#         allow_growth=True
#     )
# )
# # config = tf.ConfigProto()
# # config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))


# intermediate work area
# np.random.seed(1234)

N = 10000-1
mnist = datasets.fetch_mldata('MNIST original', data_home='.')
mnist_size = len(mnist.data)
rnd_ind = np.random.permutation(range(mnist_size))[:N]
# rnd_ind = np.random.permutation(range(mnist_size))[:mnist_size]
Y_raw = mnist.target[rnd_ind]
X = mnist.data[rnd_ind]
Y = np.eye(10)[Y_raw.astype(np.uint64)]

# random dataset selection
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# define dimensions
in_dim = np.size(X_train, 1)
out_dim = np.size(Y_train, 1)
hid_dim_lst = [200, 200, 200, 200]

learning_rate = 0.01

minibatch_size = 200
epoch_size = 20


# act_func = 'sigmoid'
act_func = 'relu'


# define layers of the model
model = Sequential()
model.add(Dense(hid_dim_lst[0], input_dim=in_dim))
model.add(Activation(act_func))

for h_dim in hid_dim_lst[1:]:
    model.add(Dense(h_dim))
    model.add(Activation(act_func))

model.add(Dense(out_dim))
model.add(Activation('softmax'))

# define loss func of the model
model.compile(
    SGD(lr=learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# model fitting
model.fit(
    X_train,
    Y_train,
    batch_size=minibatch_size,
    epochs=epoch_size
)

# model evaluation
loss_and_acc = model.evaluate(X_test, Y_test)
print()
print(loss_and_acc)


pass