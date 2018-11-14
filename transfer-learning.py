#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: d:\CodeWareHouse\Data_Science\transfer-learning.py
# Project: d:\CodeWareHouse\Data_Science
# Created Date: Saturday, July 14th 2018, 5:15:00 pm
# Author: guchenghao
# -----
# Last Modified: guchenghao
# Modified By: Saturday, 14th July 2018 7:59:18 pm
# -----
# Copyright (c) 2018 University
# Fighting!!!
# All shall be well and all shall be well and all manner of things shall be well.
# We're doomed!
###

from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import (array_to_img,
                                       img_to_array, load_img)
from keras.models import Model
from keras import backend as K
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
import numpy as np
from tqdm import tqdm
from random import shuffle
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder

train_dir = 'D:\CodeWareHouse\Data_Science\\train_wu\\'
test_dir = 'D:\CodeWareHouse\Data_Science\\test\\'
train_data = []

for imgfile in tqdm(os.listdir(train_dir)):
    if(imgfile[0] == '.'):
        pass
    else:
        label = imgfile.split('.')[0]
        image = load_img(train_dir + imgfile, target_size=(224, 224, 3))
        image_arr = img_to_array(image)
        train_data.append([image_arr, label])

shuffle(train_data)

train = train_data[:-500]
test = train_data[-500:]

train_X = np.array([item[0] for item in train]
                   ).reshape(-1, 224, 224, 3)
train_Y = np.array([item[1] for item in train])
test_X = np.array([item[0] for item in test]).reshape(-1, 224, 224, 3)
test_Y = np.array([item[1] for item in test])


le = LabelEncoder()
train_Y = le.fit_transform(train_Y)
test_Y = le.fit_transform(test_Y)

# ! 采用函数式模型
base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=(224, 224, 3))

x = base_model.output

# ! GlobalAveragePooling2D主要是用来减少训练的参数
# ! 如果使用Flatten的话，会导致模型训练的参数过多，训练结果不佳
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

cat_dog_classifier = Model(inputs=base_model.inputs, outputs=predictions)

cat_dog_classifier.summary()

# ! 将 base_model中的参数冻结
for layer in base_model.layers:
    layer.trainable = False


cat_dog_classifier.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

cat_dog_classifier.fit(train_X, train_Y, epochs=3,
                       batch_size=64, validation_data=(test_X, test_Y), verbose=1)

cat_dog_classifier.save(
    'D:\CodeWareHouse\cat_dog_classifier.h5')

preds = cat_dog_classifier.evaluate(test_X, test_Y)
print('Loss: {0}'.format(preds[0]))
print('Accuracy: {0}'.format(preds[1]))
