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

  num_paths = len(paths)

  # Twórz subplot z 2x3 kafelkami
  plt.figure(figsize=(10, num_paths*2))

  for i, image_path in enumerate(paths):
      # Wczytaj oryginalny obrazek
      original_image = Image.open(image_path)
      original_image = original_image.resize((224,224))

      plt.subplot(num_paths, 5, 5 * i + 1)
      plt.imshow(original_image)
      plt.axis('off')
      plt.title(f"Oryginalny {i+1}")

      # Konwertuj oryginalny obrazek do tablicy NumPy (multi-channel 2D image)
      original_image_np = np.array(original_image)

      # Dodaj dodatkowy wymiar, jeśli obraz jest 2D (np. szarość)
      if len(original_image_np.shape) == 2:
          original_image_np = np.expand_dims(original_image_np, axis=-1)

      # Generuj i wyświetl 4 warianty obrazu
      for j in range(4):
          augmented_image = datagen.random_transform(original_image_np)
          plt.subplot(num_paths, 5, i * 5 + j + 2)
          plt.imshow(augmented_image)
          plt.axis('off')
          plt.title(f"Wariant {i+1}-{j+1}")

  plt.tight_layout()
  plt.show()


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