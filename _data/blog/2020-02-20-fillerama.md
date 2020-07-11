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

We are surrounded by different types of smart gadgets and tech. We interact with it in our day to day life. But most of the time that interaction is just static. To make it more dynamic and user-friendly it is necessary that machines can understand how the person is feeling so that it can interact accordingly. One of the way to implement such system is to design a Computer Vision model that can identify someone's emotion using their facial expression and integrate it with existing systems or upcoming systems. Though integration is so out of scope for this post, here I will discuss how we can create a CNN model using Tensorflow Keras API, which can detect facial expressions.

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

  First of all, we will import the required libraries. 

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
