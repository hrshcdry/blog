---
template: BlogPost
path: /facial-expression-recognition
date: 2020-07-09T22:11:00.000Z
title: Facial Expression Recognition using Keras
metaDescription: >-
  There are 7 main types of facial expressions. That includes Anger, Disgust,
  Fear, Happy, Sad, Surprise, and Neutral. Detecting such expressions can be
  useful in many applications. Our aim is to develop a Facial Expression
  Recognition model using Keras.
thumbnail: /assets/2-Figure1-1.png
---
## Introduction

We are surrounded by different types of smart gadgets and tech. We interact with them in our day to day life. But most of the time that interaction is just static. To make it more dynamic and user-friendly it is necessary that machines can understand how the person is feeling so that it can interact accordingly. One of the way to implement such system is to design a Computer Vision model that can identify someone's emotion using their facial expression and integrate it with existing systems or upcoming systems. Though integration is so out of scope for this post, here I will discuss how we can create a CNN model using Tensorflow Keras API, which can detect facial expressions.

* ### Facial Expressions

  Detecting facial expressions is a good way to know how someone is feeling. Facial Expression are an accurate measure of a person's emotions unless the person is faking it. There are mainly 7 types of facial expressions:

  1. Angry
  2. Disgust
  3. Fear
  4. Happy
  5. Neutral
  6. Sad
  7. Surprise

## Creating the Model

Now we will start with the designing process of the model.

* ### Imports

  First of all, we will import the required libraries which include numpy, seaborn, matplotlib, keras, etc. If you are reading this article then I am considering you are familiar with most of these. We have imported and additional library which is livelossplot which is used to plot the loss as we train the model. We can visualize the loss in real-time.

  ```python
  import numpy as np
  import seaborn as sns
  import matplotlib.pyplot as plt
  import os
  %matplotlib inline

  from tensorflow.keras.preprocessing.image import ImageDataGenerator
  from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
  from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
  from tensorflow.keras.models import Model, Sequential
  from tensorflow.keras.optimizers import Adam
  from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
  from tensorflow.keras.utils import plot_model

  from IPython.display import SVG, Image
  from livelossplot.tf_keras import PlotLossesCallback
  import tensorflow as tf
  print("Tensorflow version:", tf.__version__)
  ```
* ### Dataset

  We will use FER2013 dataset which contains 28709 train images and 7178 test set images. The dataset was first presented in a Kaggle challenge named [Challenges in Representation Learning: Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data). You can get the dataset by clicking on the above link or from Google Drive by clicking [here](https://drive.google.com/file/d/1M-v33KWl_Tli99ShD35jpH6CP6M_OliA/view?usp=sharing). We will know more about the dataset by exploring it when we will do the preprocessing.
* ### Preprocessing

  First of all we will unzip the data as it is compressed. The zip contains two folders 'train' and 'test' and both of those folder contains 7 folders in them which named as the 7 facial expressions. Each of those 7 contains images of respective expression. After unzip, we will loop through train directory and print the number of images it contain along with respective expression. 

  ```python
  ! unzip /content/ferdataset.zip
  for expression in os.listdir("train/"):
      print(str(len(os.listdir("train/" + expression))) + " " + expression + " images")
  ```

  Output:

  ```
  3995 angry images
  4830 sad images
  4965 neutral images
  436 disgust images
  4097 fear images
  7215 happy images
  3171 surprise images
  ```

  Now, we will apply data augmentation using ImageDataGenerator which is provided by Keras. Then we will generate training and validation data from the images we are having. We are having 48x48 Greyscale images so we will pass the parameters accordingly. We have taken the batch size as 64 which is a hyperparameter we have to tune but 64 should work well on a dataset of this size. Train/ and Test/ are directories where our images are stored. 

  ```python
  img_size = 48
  batch_size = 64

  datagen_train = ImageDataGenerator(horizontal_flip=True)

  train_generator = datagen_train.flow_from_directory("train/",
                                                      target_size=(img_size,img_size),
                                                      color_mode="grayscale",
                                                      batch_size=batch_size,
                                                      class_mode='categorical',
                                                      shuffle=True)

  datagen_validation = ImageDataGenerator(horizontal_flip=True)
  validation_generator = datagen_validation.flow_from_directory("test/",
                                                      target_size=(img_size,img_size),
                                                      color_mode="grayscale",
                                                      batch_size=batch_size,
                                                      class_mode='categorical',
                                                      shuffle=False)
  ```

  Output:

  ```
  Found 28709 images belonging to 7 classes.
  Found 7178 images belonging to 7 classes.
  ```
* ### Model Architecture
