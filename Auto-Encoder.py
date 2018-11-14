import keras
import numpy as np
from keras.layers import Dense, Input
from keras.models import Model, Sequential
from keras import backend as K
from keras.datasets import mnist
from skimage import util
import matplotlib.pyplot as plt

# !由于Auto-Encoder是无监督学习，所以只取用Data部分
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
print("Minist数据的维度：{0}".format(x_train.shape))

# ! 将所有pixels全部叠成一维矩阵
x_train = x_train.reshape(x_train.shape[0], -1)
# ! 添加高斯白噪声，这样可以防止模型在训练过程中过拟合
x_train_noise = util.random_noise(x_train, seed=66)
x_test = x_test.reshape(x_test.shape[0], -1)
# ! 添加高斯白噪声
x_test_noise = util.random_noise(x_test, seed=66)
print("训练数据的维度：{0}".format(x_train.shape))

# ! 对训练数据添加噪声
# x_train_nosiy = x_train + 0.3 * \
#     np.random.normal(loc=0., scale=1., size=x_train.shape)
# x_test_nosiy = x_test + 0.3 * \
#     np.random.normal(loc=0, scale=1, size=x_test.shape)
# x_train_nosiy = np.clip(x_train_nosiy, 0., 1.)
# x_test_nosiy = np.clip(x_test_nosiy, 0, 1.)
# print(x_train_nosiy.shape, x_test_nosiy.shape)
# ! 不需要使用Keras的序贯模型
input_img = Input(shape=(784, ))

# ! 编码器
encoder = Dense(1000, activation="relu")(input_img)
encoder = Dense(500, activation="relu")(encoder)
encoder = Dense(250, activation="relu")(encoder)

# ! hidden layer，不需要设置激活函数
encoder_output = Dense(32)(encoder)

# ! 解码器
decoder = Dense(250, activation="relu")(encoder_output)
decoder = Dense(500, activation="relu")(decoder)
decoder = Dense(1000, activation="relu")(decoder)
decoder = Dense(784, activation="sigmoid")(decoder)

autoencoder = Model(input=input_img, output=decoder)

# ! 生成自编码器
encoder = Model(input=input_img, output=encoder_output)

autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.summary()


autoencoder.fit(x_train_noise, x_train, epochs=20, batch_size=128,
                verbose=1, validation_data=(x_test_noise, x_test))

# ! 预测图片
predictions = autoencoder.predict(x_test_noise)
plt.imshow(predictions[1].reshape(28, 28))
plt.gray()
plt.show()

# ! 噪声图片
plt.imshow(x_test_noise[1].reshape(28, 28))
plt.gray()
plt.show()

# ! 原图
plt.imshow(x_test[1].reshape(28, 28))
plt.gray()
plt.show()
