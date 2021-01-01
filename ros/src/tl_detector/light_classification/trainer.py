import numpy as np
import time
import cv2 
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D, MaxPool2D
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
import glob
import os 

def train(wd,hg,ch):
    folder = '/capstone/ros/imgs/'
    x=[]
    y=[]
    files = glob.glob(folder+"*.jpg")
    for f in files:
        image = cv2.imread(f)
        image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        x.append(image)
        y.append(int(f.split('/')[-1][1]))
    x = np.array(x).astype(float)
    y = np.array(y)
    n_classes = len(np.unique(y))

    model = Sequential()
    model.add(Conv2D(filters=12, kernel_size=(3, 3), activation='relu', padding="same", input_shape=(600,800,3)))
    model.add(MaxPooling2D())

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.1))

    model.add(Flatten())
    
    model.add(Dense(units=500, activation='relu'))
    model.add(Dense(units=240, activation='relu'))

    model.add(Dense(units=120, activation='relu'))

    model.add(Dense(units=84, activation='relu'))

    model.add(Dense(units=n_classes, activation = 'softmax'))
    # end lenet mod

    opt = Adam(lr=0.001)

    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model_checkpoint_callback = ModelCheckpoint("lenetmod.h5",
        save_weights_only=False,
        monitor='val_acc',
        mode='auto',
        save_best_only=True)

    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=25, verbose=1, mode='auto')
    history = model.fit(x, y, epochs=100, validation_split=0.3, callbacks=[model_checkpoint_callback, early])
train(400,400,3)