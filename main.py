from tensorflow.keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import InceptionV3, VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

from tensorflow.keras.applications import ResNet50

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report

import numpy as np
import pandas as pd
import tensorflow as tf
import random as rnd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from numpy import expand_dims

import os
import requests
import zipfile
import glob


from os import listdir
from os.path import isdir, isfile, join

from preprocessing import plot_augimages, test_classifier, plot_history, import_data, read_data

# Import and read the data
data_url = 'https://www.muratkoklu.com/datasets/vtdhnd10.php'
target_directory = 'data/'

import_data(target_directory, data_url)

data, labels = read_data(target_directory)

# Create train, test and val datasets

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)


### GENERATORS

resnet50_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.10,
    brightness_range=[0.6,1.4],
    channel_shift_range=0.7,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=preprocess_input
)

train_generator_resnet50 = resnet50_datagen.flow_from_dataframe(
        pd.DataFrame({'path': X_train, 'class': y_train}),
        x_col='path',
        y_col='class',
        target_size=(227, 227),
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
)
val_generator_resnet50 = resnet50_datagen.flow_from_dataframe(
        pd.DataFrame({'path': X_val, 'class': y_val}),
        x_col='path',
        y_col='class',
        target_size=(227, 227),
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
)
test_generator_resnet50 = resnet50_datagen.flow_from_dataframe(
        pd.DataFrame({'path': X_test, 'class': y_test}),
        x_col='path',
        y_col='class',
        target_size=(227, 227),
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
)

plot_augimages(np.random.choice(X_train, 10), resnet50_datagen)



# exclude the top (output) layer of the model
# initialize the model's weights with pre-trained weights from the ImageNet dataset
resnet50 = ResNet50(include_top = False, input_shape = (227,227,3), weights = 'imagenet')

# training of all the convolution is set to false because we'd like to use pretrained layers
for layer in resnet50.layers:
    layer.trainable = False

# the last layer of the model is removed by taking the output from the last layer
x = resnet50.layers[-1].output


x = GlobalAveragePooling2D()(x)
predictions = Dense(5, activation='softmax')(x)

# the output tensor of the pre-trained ResNet-50 model
model_resnet50 = Model(inputs = resnet50.input, outputs = predictions)

# MODEL COMPILATION
model_resnet50.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])


callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)

history_resnet50 = model_resnet50.fit(
      train_generator_resnet50,
      validation_data=val_generator_resnet50,
      epochs=50,
      callbacks=callback,
      verbose=2, #verbose=2 means that training progress will be displayed for each epoch, including metrics like loss and accuracy.
      )

plot_history(history_resnet50)

# Test model on test data
predictions = model_resnet50.predict(test_generator_resnet50)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = test_generator_resnet50.labels
print(classification_report(true_labels, predicted_labels))

