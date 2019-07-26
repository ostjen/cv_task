#!/usr/bin/env python
# coding: utf-8
import cv2
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,GlobalAveragePooling2D, Dropout
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical


#Convolutional neural network
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




if __name__ == '__main__':

  target = pd.read_csv('./train.truth.csv')
  names = target['fn']

  #get all images from the dataset(same indexes as target['fn'])
  images = []
  for name in names:
      aux = './train/' + name
      images.append(cv2.imread(aux,1))

  #encode directions 
  LE = LabelEncoder()
  target = LE.fit_transform(target['label'])
  target = np.array(target)

  #split data intro train and test
  train_x,test_x,train_y,test_y = train_test_split(images,target,test_size = 0.33)
  train_x = np.array(train_x)
  train_y = np.array(train_y)
  test_x = np.array(test_x)
  test_y = np.array(test_y)

  #one hot encoding
  train_y = to_categorical(train_y)
  test_y = to_categorical(test_y)

  train_x = train_x.astype('float32')
  train_y = train_y.astype('float32')
  test_x = test_x.astype('float32')
  test_y = test_y.astype('float32')


  #normalization of input
  train_x /= 255
  test_x /= 255

  #create model
  model = pure_cnn()
  
  model.compile(loss='categorical_crossentropy',
                optimizer=Adam(lr=0.0001), # LR = learning rate
                metrics = ['accuracy']) # Metrics to be evaluated by the model

  model_details = model.fit(train_x, train_y,
                      batch_size = 32,
                      epochs = 20, # number of iterations
                      validation_data= (test_x,test_y),
                      verbose=1)

  model.save('final_model.h5')



