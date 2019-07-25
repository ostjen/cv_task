#!/usr/bin/env python
# coding: utf-8

import cv2
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import keras.backend as K



images = [cv2.imread(file,0) for file in glob.glob("data/*.jpg")]


plt.imshow(images[2],cmap = plt.cm.gray) 


images_b = []
for i in range(len(images)):
    blur = cv2.GaussianBlur(images[i], (5, 5), 0)
    images_b.append(blur)


plt.imshow(images_b[2],cmap = plt.cm.gray)


target = pd.read_csv('./data/train.truth.csv')



LE = LabelEncoder()
target = LE.fit_transform(target['label'])
target = np.array(target)

from keras.utils import to_categorical

train_x,test_x,train_y,test_y = train_test_split(images_b,target,test_size = 0.3)

train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

train_y = to_categorical(train_y)
test_y = to_categorical(test_y)




train_x = train_x.reshape(34227,64,64,-1)
test_x = test_x.reshape(14669,64,64,-1)

#using -1 numpy infers shape

print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)




train_x = train_x.astype('float32')
train_y = train_y.astype('float32')
test_x = test_x.astype('float32')
test_y = test_y.astype('float32')


# In[18]:


train_x /= 255
test_x /= 255



model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(64,64, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(4, activation='softmax'))


from keras.optimizers import SGD

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])



model.summary()



model.fit(train_x,train_y,
          
              batch_size=32,
              epochs= 30,
              validation_data = (test_x,test_y),
              shuffle=True,
              verbose=2)


model.save('model.h5')



