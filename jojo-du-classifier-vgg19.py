# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 13:53:19 2019

@author: Priyam
"""
#Import libraries
import pandas as pd
import tensorflow as tf
import imageio
import numpy as np
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications.vgg19 import VGG19
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

#Define image dimensions
image_size = 100
image_shape = (image_size, image_size, 3)

#Define CNN model with transfer learning
base_model = VGG19(include_top=False,weights='imagenet',input_shape=image_shape,pooling=None)
base_model.trainable=False 

model = tf.keras.Sequential()
model.add(base_model)
model.add(layers.MaxPooling2D(2,padding='same'))
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(6,activation='softmax'))
#print(model.summary())

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#CSV file contains labels and image numbers 
du = pd.read_csv("resized/info.csv",header=0)
files = []
for n in range(1,len(du)+1):
    if n <10:
        filename = "000"+str(n)
    elif n<100:
        filename = "00"+str(n)
    elif n<1000:
        filename = "0"+str(n)
    filename ="resized/"+ filename + ".png"
    files.append(filename)
du["filename"]=files


def loadImages(df):         
    df = df.reset_index()
    m = len(df)
    image_list = np.empty((m,100,100,3))
    for n in range(m):
        image_path = df["filename"][n]
        im = imageio.imread(image_path)
        if im.shape == (100,100,4): #PNGs used w/ no transparency, drop 4th channel
            im=np.delete(im,3,2)
        im = im/255.0
        image_list[n]=im        
    return image_list


#Encode character names as integer values and construct dictionary for reconversion
enc = LabelEncoder() 
enc.fit(du['character'])
character_map = dict(zip(range(len(enc.classes_)),enc.classes_))

#Split dataset into training and test data
train, test = train_test_split(du,train_size=96)


training_labels=enc.transform(train['character'])

#Load training images
training_images = loadImages(train)

#Generator to augment training data with mild transformations
datagen=ImageDataGenerator(rotation_range=30,
                           shear_range=0.05,
                           zoom_range=0.1,
                           horizontal_flip=True,
                           vertical_flip=True)
datagen.fit(training_images)

es = EarlyStopping(monitor='loss',mode='min',patience=5)


model.fit(datagen.flow(training_images,training_labels,batch_size=32),
          steps_per_epoch=len(training_images),epochs=10,callbacks=[es],
          )
#model.fit(training_images,training_labels,epochs=100) #Without image augmentation

#Prepare test data in the same manner as the training data
test_labels = enc.transform(test['character'])
test_vectors = to_categorical(test_labels) #To calculate error magnitude by L2
testdata = loadImages(test)
guesses = model.predict(testdata)

#Convert predictions into a manner that's easier for error analysis
guessed = pd.Series(np.argmax(guesses,axis=-1)).map(character_map)
score = model.evaluate(testdata,test_labels)

scores = np.sqrt(np.sum(np.square(np.subtract(guesses,test_vectors)),axis=1)) #L2 error
results = pd.concat([test.reset_index(),pd.DataFrame({'guessed':guessed}),pd.DataFrame({'L2':scores})],1).drop('index',1)
score_mask = results['character']==results['guessed']
results = pd.concat([results,pd.DataFrame({'correct':score_mask})],axis=1)
