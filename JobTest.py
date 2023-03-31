# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 17:54:20 2023

@author: Albert
"""

import cv2 as cv

import numpy as np

import time

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras import layers, models

from tensorflow.keras.layers import Dense


from tensorflow.keras.models import Sequential

datasetsize = 1000;

def ConvertToGray ():
    for k in range(datasetsize):
        img = cv.imread("images/"+ str(k + 1) +"_A.jpg");
        
        for i in range(len(img)):
            for j in range(len(img[0])):
                img[i, j] = (int(img[i, j][0]) + int(img[i, j][1]) + int(img[i, j][2])) / 3; # (R + G + B) / 3
        
        cv.imwrite("grayimages/"+ str(k + 1) +".jpg", img); 
        print(k);
    
    print("Done!");
    
def ImageBinarizer ():
    for k in range(datasetsize):
        
        img = cv.imread("images/"+ str(k + 1) +"_A.jpg");
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY);
        
        binimg =cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11, 2);
        cv.imwrite("binaryimages/"+ str(k + 1) +".jpg", binimg);     
        print(k);
    
    print("done!");

#ImageBinarizer() #конвертируем полутоновое изображение в бинарное    

train_in = list(range(datasetsize));
train_out = list(range(datasetsize));


"Получение выборки и нормирование"

with open ("grayimagesf/floors.txt") as f:
        train_out = f.readlines();

for i in range(datasetsize):
    train_out[i] = int(train_out[i]) / 17; # нормирование этажности [0; 1]

for k in range(datasetsize):
    img = cv.imread("grayimagesf/"+ str(k + 1) +".jpg");
    train_in[k] = img / 255; # нормирование тональности [0; 1]


train_in = np.array(train_in);
train_out =   np.array(train_out); 

train_in = np.mean(train_in, axis=3);
print(train_in.shape);

"""-------------"""

train_in = train_in.reshape(train_in.shape[0], 256, 256, 1); 


model = models.Sequential();
model.add(layers.Conv2D(8, (5, 5), activation='relu', input_shape=(256, 256, 1)));
model.add(layers.MaxPooling2D((2, 2)));
#model.add(layers.Dropout(0.25))

model.add(layers.Flatten(input_shape=(256, 256)));
model.add(layers.Dense(32, activation='relu'));
model.add(layers.Dense(1));

model.compile(optimizer='sgd', loss=tf.keras.losses.MeanSquaredError());

t = time.time();
model.fit(train_in, train_out, batch_size=4, epochs=30);
t = time.time() - t;
print("Время обучения:");
print(t);

"""
model = Sequential();
model.add(layers.Flatten(input_shape=(256, 256)));
model.add(Dense(64, activation="sigmoid"))
model.add(Dense(1))

"задаем оптимизационную модель"
model.compile(optimizer='sgd', loss=tf.keras.losses.MeanSquaredError());


"процесс обучения"
model.fit(train_in1, train_out1, batch_size=2, epochs=50);
"""

"Тестирование"

floors = [5, 9, 9, 10];
for i in range(len(floors)):
    floors[i] = floors[i] / 17;

test = list(range(len(floors)));
for i in range(4):
    img = cv.imread("testgray/"+ str(i + 1) +".jpg");
    test[i] = img / 255;

test = np.array(test);
test = np.mean(test, axis=3);
floors = np.array(floors);

test = test.reshape(test.shape[0], 256, 256, 1);
print("Оценка отклонения подсчета для изображений вне выборки:");
loss = model.evaluate(test, floors);
print(loss);





          