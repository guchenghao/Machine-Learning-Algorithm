#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /Users/guchenghao/CodeWarehouse/Data_Science/keras_VAE.py
# Project: /Users/guchenghao/CodeWarehouse/Data_Science
# Created Date: Friday, September 14th 2018, 4:33:02 pm
# Author: Harold Gu
# -----
# Last Modified: Friday September 14th 2018 4:33:02 pm
# Modified By: Harold Gu
# -----
# Copyright (c) 2018 University
# #
# All shall be well and all shall be well and all manner of things shall be well.
# We're doomed!
###

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist

batch_size = 100  # ! 训练数据的batch大小
original_dim = 784  # ! 输入数据的维度
latent_dim = 2
intermediate_dim = 256  # ! 隐含层的神经元个数
nb_epoch = 50  # ! 训练轮数
epsilon_std = 1.0  # ! 高斯分布的标准差

# ! encoding
x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

# ! Gauss sampling, sample Z
# ! 先创建一个高斯分布, 在利用sampling出来的点进行计算


def sampling(args):
    z_mean, z_log_var = args
    # ! 生成一个标准的正态分布
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon


# note that "output_shape" isn't necessary with the TensorFlow backend
# my tips:get sample z(encoded)
# ! 自定义一个Lambda层, 并且传入一个数组
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# ! 设计VAE的decoder
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)


def vae_loss(x, x_decoded_mean):  # ! VAE的loss为 Restructtion Loss + KL loss
    # my tips:logloss
    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    # my tips:see paper's appendix B
    kl_loss = - 0.5 * K.sum(1 + z_log_var -
                            K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss


vae = Model(x, x_decoded_mean)
vae.compile(optimizer='rmsprop', loss=vae_loss)

# ! 利用minist数据集来训练VAE
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

vae.fit(x_train, x_train,
        shuffle=True,
        epochs=nb_epoch,
        verbose=2,
        batch_size=batch_size,
        validation_data=(x_test, x_test))

# ! 输出VAE的decoder的结果
# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
