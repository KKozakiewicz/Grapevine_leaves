from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from numpy import expand_dims

import os
import requests
import zipfile
import glob


from os import listdir
from os.path import isdir, isfile, join



def plot_augimages(paths, datagen):
    '''visualize augmented images generated using a Keras ImageDataGenerator'''

    plt.figure(figsize = (14,28))
    plt.suptitle('Augmented Images')

    midx = 0 # matplotlib index
    for path in paths:
        data = Image.open(path)
        data = data.resize((224,224))
        ''' add an extra dimension to the data array.
        This step is often required to make the image compatible with the batch processing
        expected by many deep learning models. The expand_dims function adds an additional
        dimension at the beginning, effectively creating a batch of size 1 containing
        the resized image.'''
        samples = expand_dims(data, 0)
        '''This line sets up an iterator (it) using a Keras ImageDataGenerator (datagen).
        The iterator is configured to generate batches of data from the samples array,
        with each batch containing one image (batch_size=1).
        This iterator can be used to generate augmented versions of the image during training.'''
        it = datagen.flow(samples, batch_size=1)

        # Show Original Image
        plt.subplot(10,5, midx+1)
        plt.imshow(data)
        plt.axis('off')

        # Show Augmented Images
        for idx, i in enumerate(range(4)):
            midx += 1
            plt.subplot(10,5, midx+1)

            batch = it.next()
            image = batch[0].astype('uint8') # extracts the image data from the batch
            plt.imshow(image)
            plt.axis('off')
        midx += 1

    plt.tight_layout()
    plt.show()


def test_classifier(
    generator,
    y_test,
    classifier,
    scoring: str = 'f1-score'
    ): # -> dict[str, Union[ClassifierMixin, float]]:

    y_pred = classifier.predict(generator)

    #  cross_val_accuracy = cross_val_score(classifier, x_train, y_train, cv=5, scoring='accuracy')
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(
        f'Model: {classifier}\n\n'
        f'Accuracy: {accuracy} \n\n',
    #   f'Cross_val_accuracy: {cross_val_accuracy} \n\n',
        f'Recall: {recall} \n\n',
        f'Precision: {precision} \n\n',
        f'f1 score: {f1}'
        )
    

def plot_history(history):
    #Plot the Loss Curves
    plt.figure(figsize=[8,6])

    plt.plot(history.history['loss'],'r',linewidth=3.0)
    plt.plot(history.history['val_loss'],'b',linewidth=3.0)

    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    plt.show()

    #Plot the Accuracy Curves
    plt.figure(figsize=[8,6])

    plt.plot(history.history['accuracy'], 'r', linewidth=3.0)
    plt.plot(history.history['val_accuracy'], 'b',linewidth=3.0)

    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)
    plt.show()


def import_data(target_directory, data_url):

    os.makedirs(target_directory, exist_ok=True)
    response = requests.get(data_url)
    zip_path = os.path.join(target_directory, 'data.zip')

    with open(zip_path, 'wb') as zip_file:
        zip_file.write(response.content)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_directory)


def read_data(target_directory):

    data_dir = os.path.join(target_directory, 'Grapevine_Leaves_Image_Dataset/*')

    data = []
    labels = []


    for class_folder in glob.glob(data_dir):
        label = os.path.basename(class_folder)

        for image_path in glob.glob(os.path.join(class_folder, '*.png')):
            labels.append(label)
            data.append(image_path)

    return data, labels