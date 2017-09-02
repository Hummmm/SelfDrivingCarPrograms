import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn

import os
import csv
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Convolution2D, Cropping2D
from keras.models import load_model
from keras import callbacks
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D

from keras.models import Model
import matplotlib.pyplot as plt
%matplotlib inline

# Crop top and bottom images to remove sky and front of the car (NVIDIA's paper)
def preprocessing(image):
    # crop to 90x320x3
    image = image[70:135,:,:]

    # scale to 160x320x3 (same as NVIDIA)
    # image = cv2.resize(image,(320, 160), interpolation = cv2.INTER_AREA)

    # convert to YUV color space (same as NVIDIA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)


def generator(samples, batch_size=16):
    num_samples = len(samples)
    correction = 0.2
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = './data/IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    measurement = float(batch_sample[3])
                    if i == 1:
                        measurement = measurement + correction
                    elif i == 2:
                        measurement = measurement - correction
                    images.append(image)
                    measurements.append(measurement)

            #Data Augmentation
            #Flipping the images
            #Multiplying the steering angle measurement with -1
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)

            #Converting the list into numpy arrays
            #This constitutes Features and Labels
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)

            yield sklearn.utils.shuffle(X_train, y_train)

#Loading CSV File
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples[1:], test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=16)
validation_generator = generator(validation_samples, batch_size=16)

#Model Architecture starts from here
model = Sequential()

#Preprocessing the images
#Normalization and Mean Centre
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))

#Image cropping
model.add(Cropping2D(cropping=((70,25),(0,0))))

#Nvidia Model starts here
model.add(Convolution2D(24,5,5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3,subsample=(2,2), activation="relu"))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator,
                                     samples_per_epoch=len(train_samples)*3,
                                     validation_data=validation_generator,
                                     nb_val_samples=len(validation_samples)*3,
                                     nb_epoch=2)

model.save('model.h5')
