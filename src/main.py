import numpy as np
#import matplotlib.pyplot as plt


#keras model stuff
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers import Conv1D, Conv2D
from keras.utils import np_utils


MAX_VALUE = 255

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)



model = Sequential()

model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

print(model.summary())



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=128, epochs=5,
          verbose=2,
          validation_data=(x_test, y_test))