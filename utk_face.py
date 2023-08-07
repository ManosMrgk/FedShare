import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential,load_model,Model
from keras.layers import Conv2D,MaxPooling2D,AvgPool2D,GlobalAveragePooling2D,Dense,Dropout,BatchNormalization,Flatten,Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input,Activation,Add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


IMG_SIZE = 100
CHANNELS_IN = 1 # 1 for grayscale 3 for color

def dataset_loader(path = "./input/UTKFace", size_x=IMG_SIZE, size_y=IMG_SIZE):
  # Load dataset
  
  pixels = []
  age = []
  gender = [] 
  i=0
  for img in os.listdir(path):
    i=i+1
    genders = img.split("_")[1]
    img = cv2.imread(str(path)+"/"+str(img))
    if CHANNELS_IN == 1:
      img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,(size_x,size_y))
    pixels.append(np.array(img))
    gender.append(np.array(genders))
  pixels = np.array(pixels)
  gender = np.array(gender,np.uint64)

  # Split Dataset into 75:25 training & test
  return train_test_split(pixels,gender,random_state=100)

def large_model(input_x=IMG_SIZE, input_y=IMG_SIZE):
  # CNN Architecture
  input = Input(shape = (input_x,input_y,CHANNELS_IN))
  conv1 = Conv2D(32,(3, 3), padding = 'same', strides=(1, 1), kernel_regularizer=l2(0.001))(input)
  conv1 = Dropout(0.1)(conv1)
  conv1 = Activation('relu')(conv1)
  pool1 = MaxPooling2D(pool_size = (2,2)) (conv1)
  conv2 = Conv2D(64,(3, 3), padding = 'same', strides=(1, 1), kernel_regularizer=l2(0.001))(pool1)
  conv2 = Dropout(0.1)(conv2)
  conv2 = Activation('relu')(conv2)
  pool2 = MaxPooling2D(pool_size = (2,2)) (conv2)
  conv3 = Conv2D(128,(3, 3), padding = 'same', strides=(1, 1), kernel_regularizer=l2(0.001))(pool2)
  conv3 = Dropout(0.1)(conv3)
  conv3 = Activation('relu')(conv3)
  pool3 = MaxPooling2D(pool_size = (2,2)) (conv3)
  conv4 = Conv2D(256,(3, 3), padding = 'same', strides=(1, 1), kernel_regularizer=l2(0.001))(pool3)
  conv4 = Dropout(0.1)(conv4)
  conv4 = Activation('relu')(conv4)
  pool4 = MaxPooling2D(pool_size = (2,2)) (conv4)
  flatten = Flatten()(pool4)
  dense_1 = Dense(128,activation='relu')(flatten)
  drop_1 = Dropout(0.2)(dense_1)
  output = Dense(2,activation="sigmoid")(drop_1)

  # Model compile
  model = Model(inputs=input,outputs=output)
  model.compile(optimizer="adam",loss=["sparse_categorical_crossentropy"],metrics=['accuracy'])
  model.summary()

  return model


# def CNN_v2(input_x=IMG_SIZE, input_y=IMG_SIZE): #Re-written in tensorflow
#     inputs = Input(shape = (input_x,input_y,CHANNELS_IN))
#     x = Conv2D(6, kernel_size=5, activation='relu')(inputs)
#     x = MaxPooling2D(pool_size=2)(x)
#     x = Conv2D(16, kernel_size=5, activation='relu')(x)
#     x = MaxPooling2D(pool_size=2)(x)
    
#     x = Flatten()(x)
#     x = Dense(120, activation='relu')(x)
#     x = Dense(84, activation='relu')(x)
#     outputs = Dense(2)(x)
    
#     model = Model(inputs=inputs, outputs=outputs)

#     # Model compile
#     model = Model(inputs=inputs,outputs=outputs)
#     model.compile(optimizer="adam",loss=["sparse_categorical_crossentropy"],metrics=['accuracy'])
#     model.summary()

#     return model


model = large_model()

# Model Checkpoint
model_path='./output/gender_model.h5'
checkpointer = ModelCheckpoint(model_path, monitor='loss',verbose=1,save_best_only=True,
                               save_weights_only=False, mode='auto',save_freq='epoch')
callback_list=[checkpointer]

x_train,x_test,y_train,y_test = dataset_loader()
save = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=30,callbacks=[callback_list])

train_loss = save.history['loss']
test_loss = save.history['val_loss']
train_accuracy = save.history['accuracy']
test_accuracy = save.history['val_accuracy']

# Plotting a line chart to visualize the loss and accuracy values by epochs.
fig, ax = plt.subplots(ncols=2, figsize=(15,7))
ax = ax.ravel()
ax[0].plot(train_loss, label='Train Loss', color='royalblue', marker='o', markersize=5)
ax[0].plot(test_loss, label='Test Loss', color = 'orangered', marker='o', markersize=5)
ax[0].set_xlabel('Epochs', fontsize=14)
ax[0].set_ylabel('Categorical Crossentropy', fontsize=14)
ax[0].legend(fontsize=14)
ax[0].tick_params(axis='both', labelsize=12)
ax[1].plot(train_accuracy, label='Train Accuracy', color='royalblue', marker='o', markersize=5)
ax[1].plot(test_accuracy, label='Test Accuracy', color='orangered', marker='o', markersize=5)
ax[1].set_xlabel('Epochs', fontsize=14)
ax[1].set_ylabel('Accuracy', fontsize=14)
ax[1].legend(fontsize=14)
ax[1].tick_params(axis='both', labelsize=12)
fig.suptitle(x=0.5, y=0.92, t="Lineplots showing loss and accuracy of CNN model by epochs", fontsize=16)
