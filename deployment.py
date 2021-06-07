from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
np.random.seed(2)
train=ImageDataGenerator(rescale=1. /255,rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test=ImageDataGenerator(rescale=1./255)
train_dataset=train.flow_from_directory('C:\Users\MOUNIK\Desktop\FakeCurrencyDetection\Traindatasetall',target_size=(224,224),batch_size=32,class_mode='binary')
test_dataset=test.flow_from_directory('C:\Users\MOUNIK\Desktop\FakeCurrencyDetection\Testdatasetall',target_size=(224,224),batch_size=32,class_mode='binary')
model=keras.Sequential()
model=tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(224,224,3)),
                                  tf.keras.layers.MaxPool2D(2,2),
                                  tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)),
                                  tf.keras.layers.MaxPool2D(2,2),
                                  tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(224,224,3)),
                                  tf.keras.layers.MaxPool2D(2,2),
                                  tf.keras.layers.Flatten(),
                                  tf.keras.layers.Dense(512,activation='relu'),
                                  tf.keras.layers.Dense(1,activation='sigmoid')
                                  ])
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
model_fit=model.fit(train_dataset,steps_per_epoch=11,epochs=100,validation_data=test_dataset)
model.load_weights('C:\Users\MOUNIK\Desktop\FakeCurrencyDetection\se1.h5')
