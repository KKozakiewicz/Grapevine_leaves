from tensorflow.keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

from tensorflow.keras.applications import ResNet50

from sklearn.model_selection import train_test_split
from sklearn.metrics import  classification_report

import numpy as np
import pandas as pd
import tensorflow as tf



from preprocessing import plot_augimages, plot_history, import_data, read_data

# Import and read the data
data_url = 'https://www.muratkoklu.com/datasets/vtdhnd10.php'
target_directory = 'data/'

import_data(target_directory, data_url)

data, labels = read_data(target_directory)
df = pd.DataFrame({'path': data, 'class': labels})

# Create train, test and val datasets

X_train, X_test, y_train, y_test = train_test_split(df['path'], df['class'], test_size=0.1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.11)

df_train = pd.concat([X_train, y_train], axis =1)
df_val = pd.concat([X_val, y_val], axis =1)
df_test = pd.concat([X_test, y_test], axis =1)


### GENERATORS

resnet50_train_datagen = ImageDataGenerator(
    rotation_range=90,
    zoom_range=0.10,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

resnet50_datagen = ImageDataGenerator(preprocessing_function = tf.image.rgb_to_grayscale)

train_generator_resnet50 = resnet50_train_datagen.flow_from_dataframe(
        df_train,
        #color_mode = 'grayscale',
        x_col='path',
        y_col='class',
        target_size=(227, 227),
        batch_size=128,
        class_mode="categorical",
        shuffle=True,
)
val_generator_resnet50 = resnet50_datagen.flow_from_dataframe( #flow_from_directory
        df_val,
        #color_mode = 'grayscale',
        x_col='path',
        y_col='class',
        target_size=(227, 227),
        batch_size=128,
        class_mode="categorical",
        shuffle=True,
)
test_generator_resnet50 = resnet50_datagen.flow_from_dataframe(
        df_test,
        #color_mode = 'grayscale',
        x_col='path',
        y_col='class',
        target_size=(227, 227),
        batch_size=128,
        class_mode="categorical",
        shuffle=False,
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
x = Dropout(0.2)(x)
predictions = Dense(5, activation='softmax')(x)

# the output tensor of the pre-trained ResNet-50 model
model_resnet50 = Model(inputs = resnet50.input, outputs = predictions)

model_resnet50.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)

history_resnet50 = model_resnet50.fit(
      train_generator_resnet50,
      validation_data=val_generator_resnet50,
      epochs=50,
      callbacks=callback,
      verbose=2,
      )

plot_history(history_resnet50)

# Display model statistics
print(classification_report(test_generator_resnet50.labels, predictions))

