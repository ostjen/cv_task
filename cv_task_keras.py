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
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D, Lambda, Dropout
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import keras.backend as K

target = pd.read_csv('./train.truth.csv')
names = target['fn']

images = []
for name in names:
    aux = './data/' + name
    images.append(cv2.imread(aux,1))


LE = LabelEncoder()
target = LE.fit_transform(target['label'])
target = np.array(target)

from keras.utils import to_categorical

train_x,test_x,train_y,test_y = train_test_split(images,target,test_size = 0.33)

train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

train_y = to_categorical(train_y)
test_y = to_categorical(test_y)


#train_x = train_x.reshape(34227,64,64,3)
#test_x = test_x.reshape(14669,64,64,3)

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


'''
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(64,64, 1)))
model.add(Dropout(0.50))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(0.50))
model.add(Dense(4, activation='softmax'))


from keras.optimizers import SGD

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
'''
# Model creation


def pure_cnn():
    
    model = Sequential()
    
    model.add(Conv2D(96, (3, 3), activation='relu', padding = 'same', input_shape=(64,64,3)))    
    model.add(Dropout(0.2))
    
    model.add(Conv2D(96, (3, 3), activation='relu', padding = 'same'))  
    model.add(Conv2D(96, (3, 3), activation='relu', padding = 'same', strides = 2))    
    model.add(Dropout(0.5))
    
    model.add(Conv2D(192, (3, 3), activation='relu', padding = 'same'))    
    model.add(Conv2D(192, (3, 3), activation='relu', padding = 'same'))
    model.add(Conv2D(192, (3, 3), activation='relu', padding = 'same', strides = 2))    
    model.add(Dropout(0.5))    
    
    model.add(Conv2D(192, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (1, 1),padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(4, (1, 1), padding='valid'))

    model.add(GlobalAveragePooling2D())
    
    model.add(Activation('softmax'))

    model.summary()
    
    return model

from keras.optimizers import Adam, SGD
model = pure_cnn()
model.compile(loss='categorical_crossentropy', # Better loss function for neural networks
              optimizer=Adam(lr=0.0001), # Adam optimizer with 1.0e-4 learning rate
              metrics = ['accuracy']) # Metrics to be evaluated by the model

model_details = model.fit(train_x, train_y,
                    batch_size = 32,
                    epochs = 20, # number of iterations
                    validation_data= (test_x,test_y),
                    verbose=1)

model.save('model_nogaussian.h5')



