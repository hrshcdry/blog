---
template: BlogPost
path: /face-mask-detection
date: 2020-10-05T15:05:49.332Z
title: Facemask Detection from Videos
metaDescription: >-
  Here, I have tried to automate the process of detecting people who not wear a
  mask in public places from the videos. It can be used with CCTV to monitor and
  control the spread of the virus.
thumbnail: /assets/ezgif.com-crop.jpg
---
## Introduction

In such difficult times, it has become more important than ever for people to wear masks in public places. Though most people wear masks, there exist some who just don't take enough precautions even after a lot of guidelines from governments and other people. For this reason, in public places like offices, stations, etc they had to employ people to monitor such activities. In this project, I tried to automate the process using Convolutional Neural Network(CNN) and OpenCV.  It is deployed using the Flask framework. This system helps reduce the human effort required for such a task. 

In the development of a pipeline for this system, there are mainly 5 steps involved:

1. Extracting Faces using pretrained dnn model using OpenCV.
2. Train a CNN model to detect whether those faces are masked or not.
3. Combining both and use it to detect masked faces from a single image.
4. Use that for the detection in video frames.
5. Deploy the model using the Flask framework.

### 1. Extracting faces from an Image

We have used opencv with a pretrained singleshot resnet based deep neural network to extract the faces available in an image. We will load the model alongwith weights. Then, will pass the address of an image and preprocess it using blogFromImage. In the end we will pass the processed image from the model we loaded. Below is the code for the same:

```python
import cv2
import os

prototxtPath = os.path.sep.join(["/content", "deploy.prototxt"])
weightsPath = os.path.join("/content",
	"res10_300x300_ssd_iter_140000.caffemodel")
net = cv2.dnn.readNet(prototxtPath, weightsPath)

image = cv2.imread("/content/2020_4$largeimg_1457131882.jpg")
orig = image.copy()
(h, w) = image.shape[:2]

# construct a blob from the image
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
	(104.0, 177.0, 123.0))

# pass the blob through the network and obtain the face detections
print("computing face detections...")
net.setInput(blob)
detections = net.forward()
```

The variable "detections" at the end contains the coordinates of the faces available in the image.

## 2. Train the CNN Model for detecting masks

We have tried two different models, first being my custom model and for second we have retrained Mobilenet with Imagenet weights. As our dataset was too small, custom model ended up giving  98.10% accuracy. We know that transfer learning is having a great track record when it comes to small datasets. So, we tried retraining Mobilenet. We first loaded mobilenet with imagenet weights and trained it for 20 epochs by changing last dense layers. Here is the code for the same:

### Mobilenet:

```python
base1 = MobileNetV2(weights="imagenet", include_top=False,
	input_shape=(224, 224, 3))

head1 = base1.output
head1 = AveragePooling2D(pool_size=(7, 7))(head1)
head1 = Flatten(name="flatten")(head1)
head1 = Dense(128, activation="relu")(head1)
head1 = Dropout(0.5)(head1)
head1 = Dense(2, activation="softmax")(head1)
model1 = Model(inputs=base1.input, outputs=head1)

for layer in base1.layers:
	layer.trainable = False

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model1.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
H = model1.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)
```

Output:

```
Epoch 1/20
34/34 [==============================] - 11s 316ms/step - loss: 0.4591 - accuracy: 0.7949 - val_loss: 0.0881 - val_accuracy: 0.9783
Epoch 2/20
34/34 [==============================] - 10s 299ms/step - loss: 0.1065 - accuracy: 0.9642 - val_loss: 0.0472 - val_accuracy: 0.9855
Epoch 3/20
34/34 [==============================] - 10s 297ms/step - loss: 0.0683 - accuracy: 0.9794 - val_loss: 0.0367 - val_accuracy: 0.9891
Epoch 4/20
34/34 [==============================] - 10s 294ms/step - loss: 0.0550 - accuracy: 0.9878 - val_loss: 0.0299 - val_accuracy: 0.9928
Epoch 5/20
34/34 [==============================] - 10s 296ms/step - loss: 0.0414 - accuracy: 0.9888 - val_loss: 0.0274 - val_accuracy: 0.9928
Epoch 6/20
34/34 [==============================] - 10s 296ms/step - loss: 0.0337 - accuracy: 0.9934 - val_loss: 0.0226 - val_accuracy: 0.9928
Epoch 7/20
34/34 [==============================] - 10s 295ms/step - loss: 0.0281 - accuracy: 0.9906 - val_loss: 0.0212 - val_accuracy: 0.9928
Epoch 8/20
34/34 [==============================] - 10s 294ms/step - loss: 0.0339 - accuracy: 0.9869 - val_loss: 0.0204 - val_accuracy: 0.9928
Epoch 9/20
34/34 [==============================] - 10s 293ms/step - loss: 0.0251 - accuracy: 0.9934 - val_loss: 0.0197 - val_accuracy: 0.9928
Epoch 10/20
34/34 [==============================] - 10s 300ms/step - loss: 0.0223 - accuracy: 0.9934 - val_loss: 0.0171 - val_accuracy: 0.9928
Epoch 11/20
34/34 [==============================] - 10s 297ms/step - loss: 0.0281 - accuracy: 0.9897 - val_loss: 0.0172 - val_accuracy: 0.9928
Epoch 12/20
34/34 [==============================] - 10s 293ms/step - loss: 0.0260 - accuracy: 0.9916 - val_loss: 0.0152 - val_accuracy: 0.9928
Epoch 13/20
34/34 [==============================] - 10s 293ms/step - loss: 0.0137 - accuracy: 0.9963 - val_loss: 0.0166 - val_accuracy: 0.9928
Epoch 14/20
34/34 [==============================] - 10s 294ms/step - loss: 0.0174 - accuracy: 0.9934 - val_loss: 0.0158 - val_accuracy: 0.9928
Epoch 15/20
34/34 [==============================] - 10s 294ms/step - loss: 0.0207 - accuracy: 0.9916 - val_loss: 0.0168 - val_accuracy: 0.9928
Epoch 16/20
34/34 [==============================] - 10s 294ms/step - loss: 0.0154 - accuracy: 0.9963 - val_loss: 0.0176 - val_accuracy: 0.9928
Epoch 17/20
34/34 [==============================] - 10s 295ms/step - loss: 0.0105 - accuracy: 0.9981 - val_loss: 0.0178 - val_accuracy: 0.9928
Epoch 18/20
34/34 [==============================] - 10s 295ms/step - loss: 0.0138 - accuracy: 0.9963 - val_loss: 0.0179 - val_accuracy: 0.9928
Epoch 19/20
34/34 [==============================] - 10s 298ms/step - loss: 0.0119 - accuracy: 0.9972 - val_loss: 0.0174 - val_accuracy: 0.9928
Epoch 20/20
34/34 [==============================] - 10s 294ms/step - loss: 0.0092 - accuracy: 0.9963 - val_loss: 0.0173 - val_accuracy: 0.9928
```

### Custom Model:

```python
input2 = Input(shape=(224,224,3))  

# 1st Conv
hidden2 = Conv2D(64, (3,3), padding="valid")(input2)
hidden2 = BatchNormalization()(hidden2)
hidden2 = Activation('relu')(hidden2)
hidden2 = MaxPooling2D(pool_size=(2, 2))(hidden2)
hidden2 = Dropout(0.25)(hidden2)

# 2nd Conv
hidden2 = Conv2D(128, (5,5), padding="valid")(hidden2)
hidden2 = BatchNormalization()(hidden2)
hidden2 = Activation('relu')(hidden2)
hidden2 = MaxPooling2D(pool_size=(2, 2))(hidden2)
hidden2 = Dropout(0.25)(hidden2)

# Flattening
hidden2 = Flatten()(hidden2)

# Fully connected layer layer
hidden2 = Dense(256)(hidden2)
hidden2 = BatchNormalization()(hidden2)
hidden2 = Activation('relu')(hidden2)
hidden2 = Dropout(0.25)(hidden2)

output2 = Dense(2, activation='softmax')(hidden2)

# Create the model
model2 = Model(inputs=input2, outputs=output2)

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model2.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
model2.summary()
```

Output:

```
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         [(None, 224, 224, 3)]     0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 222, 222, 64)      1792      
_________________________________________________________________
batch_normalization_3 (Batch (None, 222, 222, 64)      256       
_________________________________________________________________
activation_3 (Activation)    (None, 222, 222, 64)      0         
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 111, 111, 64)      0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 111, 111, 64)      0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 107, 107, 128)     204928    
_________________________________________________________________
batch_normalization_4 (Batch (None, 107, 107, 128)     512       
_________________________________________________________________
activation_4 (Activation)    (None, 107, 107, 128)     0         
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 53, 53, 128)       0         
_________________________________________________________________
dropout_4 (Dropout)          (None, 53, 53, 128)       0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 359552)            0         
_________________________________________________________________
dense_2 (Dense)              (None, 256)               92045568  
_________________________________________________________________
batch_normalization_5 (Batch (None, 256)               1024      
_________________________________________________________________
activation_5 (Activation)    (None, 256)               0         
_________________________________________________________________
dropout_5 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 514       
=================================================================
Total params: 92,254,594
Trainable params: 92,253,698
Non-trainable params: 896
_________________________________________________________________
```

```python
H2 = model2.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)
```

```
Epoch 1/20
34/34 [==============================] - 13s 371ms/step - loss: 0.6355 - accuracy: 0.8202 - val_loss: 0.3318 - val_accuracy: 0.8478
Epoch 2/20
34/34 [==============================] - 12s 366ms/step - loss: 0.3237 - accuracy: 0.9017 - val_loss: 0.4986 - val_accuracy: 0.6993
Epoch 3/20
34/34 [==============================] - 13s 372ms/step - loss: 0.2265 - accuracy: 0.9223 - val_loss: 0.9061 - val_accuracy: 0.5000
Epoch 4/20
34/34 [==============================] - 12s 364ms/step - loss: 0.1959 - accuracy: 0.9354 - val_loss: 0.6262 - val_accuracy: 0.5652
Epoch 5/20
34/34 [==============================] - 12s 363ms/step - loss: 0.2166 - accuracy: 0.9232 - val_loss: 0.7886 - val_accuracy: 0.5036
Epoch 6/20
34/34 [==============================] - 12s 361ms/step - loss: 0.1406 - accuracy: 0.9476 - val_loss: 0.9089 - val_accuracy: 0.5145
Epoch 7/20
34/34 [==============================] - 12s 362ms/step - loss: 0.1315 - accuracy: 0.9494 - val_loss: 0.8267 - val_accuracy: 0.5109
Epoch 8/20
34/34 [==============================] - 12s 364ms/step - loss: 0.1279 - accuracy: 0.9541 - val_loss: 0.7412 - val_accuracy: 0.5616
Epoch 9/20
34/34 [==============================] - 12s 365ms/step - loss: 0.1336 - accuracy: 0.9457 - val_loss: 0.6042 - val_accuracy: 0.6413
Epoch 10/20
34/34 [==============================] - 13s 368ms/step - loss: 0.1396 - accuracy: 0.9485 - val_loss: 0.5673 - val_accuracy: 0.6775
Epoch 11/20
34/34 [==============================] - 12s 365ms/step - loss: 0.1380 - accuracy: 0.9494 - val_loss: 0.8537 - val_accuracy: 0.5688
Epoch 12/20
34/34 [==============================] - 12s 364ms/step - loss: 0.1033 - accuracy: 0.9654 - val_loss: 0.2622 - val_accuracy: 0.8841
Epoch 13/20
34/34 [==============================] - 13s 372ms/step - loss: 0.1177 - accuracy: 0.9504 - val_loss: 0.3416 - val_accuracy: 0.8333
Epoch 14/20
34/34 [==============================] - 12s 365ms/step - loss: 0.0981 - accuracy: 0.9625 - val_loss: 0.1896 - val_accuracy: 0.9275
Epoch 15/20
34/34 [==============================] - 12s 365ms/step - loss: 0.1364 - accuracy: 0.9551 - val_loss: 0.2032 - val_accuracy: 0.9058
Epoch 16/20
34/34 [==============================] - 12s 365ms/step - loss: 0.1238 - accuracy: 0.9522 - val_loss: 0.0856 - val_accuracy: 0.9746
Epoch 17/20
34/34 [==============================] - 12s 363ms/step - loss: 0.1045 - accuracy: 0.9625 - val_loss: 0.0883 - val_accuracy: 0.9746
Epoch 18/20
34/34 [==============================] - 12s 361ms/step - loss: 0.0702 - accuracy: 0.9719 - val_loss: 0.0899 - val_accuracy: 0.9674
Epoch 19/20
34/34 [==============================] - 12s 367ms/step - loss: 0.0853 - accuracy: 0.9691 - val_loss: 0.0812 - val_accuracy: 0.9746
Epoch 20/20
34/34 [==============================] - 12s 363ms/step - loss: 0.0748 - accuracy: 0.9682 - val_loss: 0.0516 - val_accuracy: 0.9819
```

### Accuracy

#### Custom Model

```
              precision    recall  f1-score   support

   with_mask       0.99      0.97      0.98       138
without_mask       0.97      0.99      0.98       138

    accuracy                           0.98       276
   macro avg       0.98      0.98      0.98       276
weighted avg       0.98      0.98      0.98       276
```



#### Transfer Learning using Mobilenet

```
              precision    recall  f1-score   support

   with_mask       0.99      0.99      0.99       138
without_mask       0.99      0.99      0.99       138

    accuracy                           0.99       276
   macro avg       0.99      0.99      0.99       276
weighted avg       0.99      0.99      0.99       276
```

## 3. Combining Both to Get Location of faces alongwith mask detection

We then combined both and created a pipeline where if we give an image it first extracts all the faces and then passes all the faces from the CNN model we trained in a loop to detect which ones have masks on and which haven't. In the end, it draws red and gree bounding boxes around it which refer to without_mask and with_mask labels respectively. 

Below is the code and the output:

```python
for i in range(0, detections.shape[2]):
	confidence = detections[0, 0, i, 2]

	# .5 confidence filter
	if confidence > 0.5:
		# coordinates of the bounding box 
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# check that bounding box fall within the dimensions of frame
		(startX, startY) = (max(0, startX), max(0, startY))
		(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

		# processing
		face = image[startY:endY, startX:endX]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)
		face = np.expand_dims(face, axis=0)

		# passing to mask detector
		(mask, withoutMask) = model.predict(face)[0]

		# bouding box stuff
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# probability
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box
		cv2.putText(image, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
```

We got the following output when we passed an example image and then printed the variable "image".

![](/assets/download (2).png)

## 4. Detecting from Videos (By Extracting frames and passing through the previous pipeline in a loop)
