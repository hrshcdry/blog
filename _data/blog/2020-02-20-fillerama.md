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

  Now our data is ready so we will create the model and then pass data.
* ### Model Architecture

  We will be implementing the following model is Keras which was proposed in the paper FER13:

![](https://github.com/hrshcdry/Facial_Expression_Recognition/blob/master/model.png?raw=true "Model Architecture")

As shown in image above we will implement 4 Convolutional Layers, 2 Fully Connected Layers. and an output layer with size=7 which is having a softmax activation.

```python
# Initialising the CNN
model = Sequential()

# 1st Conv Layer
model.add(Conv2D(64,(3,3), padding='same', input_shape=(48, 48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 2nd Conv layer
model.add(Conv2D(128,(5,5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 3rd Conv layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 4th Conv layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flattening
model.add(Flatten())

# 1st FCN
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

# 2nd FCN
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(7, activation='softmax'))

opt = Adam(lr=0.0005)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

Output:

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 48, 48, 64)        640       
_________________________________________________________________
batch_normalization (BatchNo (None, 48, 48, 64)        256       
_________________________________________________________________
activation (Activation)      (None, 48, 48, 64)        0         
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 24, 24, 64)        0         
_________________________________________________________________
dropout (Dropout)            (None, 24, 24, 64)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 24, 24, 128)       204928    
_________________________________________________________________
batch_normalization_1 (Batch (None, 24, 24, 128)       512       
_________________________________________________________________
activation_1 (Activation)    (None, 24, 24, 128)       0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 12, 12, 128)       0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 12, 12, 128)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 12, 12, 512)       590336    
_________________________________________________________________
batch_normalization_2 (Batch (None, 12, 12, 512)       2048      
_________________________________________________________________
activation_2 (Activation)    (None, 12, 12, 512)       0         
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 6, 6, 512)         0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 6, 6, 512)         0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 6, 6, 512)         2359808   
_________________________________________________________________
batch_normalization_3 (Batch (None, 6, 6, 512)         2048      
_________________________________________________________________
activation_3 (Activation)    (None, 6, 6, 512)         0         
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 3, 3, 512)         0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 3, 3, 512)         0         
_________________________________________________________________
flatten (Flatten)            (None, 4608)              0         
_________________________________________________________________
dense (Dense)                (None, 256)               1179904   
_________________________________________________________________
batch_normalization_4 (Batch (None, 256)               1024      
_________________________________________________________________
activation_4 (Activation)    (None, 256)               0         
_________________________________________________________________
dropout_4 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 512)               131584    
_________________________________________________________________
batch_normalization_5 (Batch (None, 512)               2048      
_________________________________________________________________
activation_5 (Activation)    (None, 512)               0         
_________________________________________________________________
dropout_5 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 7)                 3591      
=================================================================
Total params: 4,478,727
Trainable params: 4,474,759
Non-trainable params: 3,968
_________________________________________________________________
```

After designing the model we have compiled it. For loss function we have used Categorial Crossentropy and Adam is used as optimizer as it works best most of the time. In the end we have called *model.summary(),* which prints the summary of the model. In output we can see the summary which shows all the layers, their respective output shapes and number of parameters. Now we will move to the training section.

* ### Model Training

  ```python
  %%time

  epochs = 20
  steps_per_epoch = train_generator.n//train_generator.batch_size
  validation_steps = validation_generator.n//validation_generator.batch_size

  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                patience=2, min_lr=0.00001, mode='auto')
  checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_accuracy',
                               save_weights_only=True, mode='max', verbose=1)
  callbacks = [checkpoint, reduce_lr]

  history = model.fit(
      x=train_generator,
      steps_per_epoch=steps_per_epoch,
      epochs=epochs,
      validation_data = validation_generator,
      validation_steps = validation_steps,
      callbacks=callbacks
  )
  ```

  Ouptput:

  Original output is too big so I have only shown the first and last few epochs only. 

  ```
  Epoch 1/20
  448/448 [==============================] - ETA: 0s - loss: 1.8055 - accuracy: 0.3056
  Epoch 00001: saving model to model_weights.h5
  448/448 [==============================] - 124s 277ms/step - loss: 1.8055 - accuracy: 0.3056 - val_loss: 1.7709 - val_accuracy: 0.3627 - lr: 5.0000e-04
  Epoch 2/20
  448/448 [==============================] - ETA: 0s - loss: 1.4822 - accuracy: 0.4320
  Epoch 00002: saving model to model_weights.h5
  448/448 [==============================] - 125s 279ms/step - loss: 1.4822 - accuracy: 0.4320 - val_loss: 1.3154 - val_accuracy: 0.4901 - lr: 5.0000e-04
  Epoch 3/20
  448/448 [==============================] - ETA: 0s - loss: 1.3252 - accuracy: 0.4908
  Epoch 00003: saving model to model_weights.h5
  448/448 [==============================] - 125s 279ms/step - loss: 1.3252 - accuracy: 0.4908 - val_loss: 1.4431 - val_accuracy: 0.4707 - lr: 5.0000e-04
  .
  .
  .
  Epoch 18/20
  448/448 [==============================] - ETA: 0s - loss: 0.8328 - accuracy: 0.6891
  Epoch 00018: saving model to model_weights.h5
  448/448 [==============================] - 126s 281ms/step - loss: 0.8328 - accuracy: 0.6891 - val_loss: 0.9688 - val_accuracy: 0.6445 - lr: 5.0000e-05
  Epoch 19/20
  448/448 [==============================] - ETA: 0s - loss: 0.8178 - accuracy: 0.6955
  Epoch 00019: saving model to model_weights.h5
  448/448 [==============================] - 125s 279ms/step - loss: 0.8178 - accuracy: 0.6955 - val_loss: 0.9579 - val_accuracy: 0.6498 - lr: 1.0000e-05
  Epoch 20/20
  448/448 [==============================] - ETA: 0s - loss: 0.8176 - accuracy: 0.6970
  Epoch 00020: saving model to model_weights.h5
  448/448 [==============================] - 125s 278ms/step - loss: 0.8176 - accuracy: 0.6970 - val_loss: 0.9549 - val_accuracy: 0.6547 - lr: 1.0000e-05
  ```

  I first trained it with 15 epochs but after training I realized that loss was still decreasing so I trained it with another 5 epochs so total 20. After 20 the loss did not seem improving so stopped at 20. We used learning rate decay here and saved weights on every epochs. So, that we can use the final weights to predict when we create a full system.

  ```python
  model_json = model.to_json()
  with open("model.json", "w") as json_file:
      json_file.write(model_json)
  ```

  The above code exports the model as a JSON file, which we will also use for making prediction.

  After 20 epochs, we got 69.7% training accuracy and 65.47% testing accuracy which is not state of the art but considering it is a 7 class classification, it can be called a pretty decent accuracy.

## Conclusion

We successfully trained a  model that can identify facial expression with a decent amount of accuracy. We can further improve the model by fine-tuning existing state of the art models like VGG-16, Resnet, etc. Which will try to do in the future. We have also implemented a flask application which can detect such expressions directly from webcam or a video stream. We will cover that topic in a separate post as it is out of the scope for this one. Though if you want to explore it, the code can be found on my [Github](https://github.com/hrshcdry/Facial_Expression_Recognition).

If you have any idea feel free to write me at [noob@harsh.codes](mailto:noob@harsh.codes).
