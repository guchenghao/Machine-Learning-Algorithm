#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: d:\CodeWareHouse\Data_Science\AlexNet.py
# Project: d:\CodeWareHouse\Data_Science
# Created Date: Tuesday, July 24th 2018, 9:30:35 pm
# Author: guchenghao
# -----
# Last Modified: guchenghao
# Modified By: Sunday, 29th July 2018 7:26:49 pm
# -----
# Copyright (c) 2018 University
# Fighting!!!
# All shall be well and all shall be well and all manner of things shall be well.
# We're doomed!
###
import keras
import numpy as np
from keras.layers import Dense, Input, Conv2D, MaxPool2D, Flatten, Dropout
from keras.models import Model, Sequential
from keras import backend as K
from keras.datasets import mnist

import matplotlib.pyplot as plt

# (x_train, y_train), (x_test, y_test) = mnist.load_data()

input_image = Input(shape=(227, 227, 3))


conv1 = Conv2D(filters=96, kernel_size=11, strides=(
    4, 4), activation='relu')(input_image)
maxpool1 = MaxPool2D(pool_size=(3, 3), strides=(
    2, 2))(conv1)


conv2 = Conv2D(filters=256, kernel_size=5, strides=(
    1, 1), activation='relu', padding='same')(maxpool1)
maxpool2 = MaxPool2D(pool_size=(3, 3), strides=(
    2, 2))(conv2)


conv3 = Conv2D(filters=384, kernel_size=3, strides=(
    1, 1), activation='relu', padding='same')(maxpool2)


conv4 = Conv2D(filters=384, kernel_size=3, strides=(
    1, 1), activation='relu', padding='same')(conv3)


conv5 = Conv2D(filters=256, kernel_size=3, strides=(
    1, 1), activation='relu', padding='same')(conv4)
maxpool3 = MaxPool2D(pool_size=(3, 3), strides=(
    2, 2))(conv5)

Fcn = Flatten()(maxpool3)

f_layer_1 = Dense(units=4096, activation='relu')(Fcn)
f_layer_1 = Dropout(0.5)(f_layer_1)

f_layer_2 = Dense(units=4096, activation='relu')(f_layer_1)
f_layer_2 = Dropout(0.5)(f_layer_2)

final_layer = Dense(1000, activation='softmax')(f_layer_2)


minist_AlexNet = Model(inputs=input_image, outputs=final_layer)

minist_AlexNet.summary()

# minist_AlexNet.compile(
#     optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])


# cat_dog_classifier.fit(x_train, y_train, epochs=3,
#                        batch_size=64, validation_data=(x_test, y_test), verbose=1)
