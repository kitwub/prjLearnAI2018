from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

import numpy as np

np.random.seed(0)  # 乱数シード

M = 2
K = 1
H = 2
Epoch_num = 5000

X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0],[1],[1],[0]])

model = Sequential()
model.add(Dense(units=H))
# model.add(Dense(input_dim=M,units=H))
model.add(Activation('sigmoid'))
model.add(Dense(units=K))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))

model.fit(X, Y, batch_size=4, epochs=Epoch_num)

prediction = model.predict_classes(X)
prob = model.predict_proba(X)

print(prediction)
print(prob)

pass
