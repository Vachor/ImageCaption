from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Input
from keras.models import Model
from keras.utils import to_categorical
import numpy as np
from keras.datasets import cifar100,mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import Adam
import cv2
import matplotlib.pyplot as plt
import pickle
import os

with open(os.path.join('cifar','x_train_o.pkl'),'rb') as f:
    x_train_o = pickle.load(f)
with open(os.path.join('cifar','y_train.pkl'),'rb') as f:
    y_train = pickle.load(f)
with open(os.path.join('cifar','x_test_o.pkl'),'rb') as f:
    x_test_o = pickle.load(f)
with open(os.path.join('cifar','y_test.pkl'),'rb') as f:
    y_test = pickle.load(f)


x_train = np.zeros([10000,224,224,3])
x_test = np.zeros([10000,224,224,3])
for i,value in enumerate(x_train_o):
    if i >= 10000:
        break
    x_train[i] = cv2.resize(x_train_o[i],(224,224))
    x_test[i] = cv2.resize(x_test_o[i],(224,224))

y_train = np_utils.to_categorical(y_train, num_classes=100)
y_test = np_utils.to_categorical(y_test, num_classes=100)
input = Input(shape=(224,224,3))
conv1 = Convolution2D(32,(3,3),activation='relu',padding='same')(input)
conv2 = conv1
maxpool1 = MaxPooling2D(pool_size=(2,2))(conv2)
conv3 = Convolution2D(64,(3,3),activation='relu',padding='same')(maxpool1)
conv4 = Convolution2D(64,(3,3),activation='relu',padding='same')(conv3)
maxpool2 = MaxPooling2D(pool_size=(2,2))(conv4)
flaten = Flatten()(maxpool2)
dense1 = Dense(1028,activation='relu')(flaten)
dense2 = Dense(100,activation='softmax')(dense1)
model = Model(inputs = input, outputs=dense2)
print(model.summary())
# model.add(Convolution2D(filters=32,
#                         kernel_size=(5,5),
#                         activation='relu',
#                         input_shape=(224,224,3)))
# model.add(Convolution2D(filters=64,
#                         kernel_size=(5,5),
#                         activation='relu'))
#
# model.add(MaxPooling2D(pool_size=(2,2)))
#
# model.add(Convolution2D(filters=128,
#                         kernel_size=(5,5),
#                         activation='relu'))
# model.add(Convolution2D(filters=128,
#                         kernel_size=(5,5),
#                         activation='relu'))
#
# model.add(MaxPooling2D(pool_size=(2,2)))
#
# model.add(Flatten())
#
# model.add(Dense(1024,activation='relu'))
#
# model.add(Dense(100,activation='softmax'))

# adam = Adam(lr=1e-4)
model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])
print('Training -------------')
model.fit(x_train,y_train[0:10000],epochs=10,batch_size=64)
print('\nTesting -----------')
loss,accuracy = model.evaluate(x_test,y_test)

print('\ntest loss: ',loss)
print('\ntest accuracy: ',accuracy)
model.save(os.path.join('CNNMODELS','baseline_v2.h5'))