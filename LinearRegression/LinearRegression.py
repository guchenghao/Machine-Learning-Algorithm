# @Author: guchenghao
# @Date:   2017-04-26T18:32:29+08:00
# @Email:  1754926878@qq.com
# @Last modified by:   guchenghao
# @Last modified time: 2017-05-21T17:46:23+08:00


import numpy as np

from plotData import plotDataAndDrawLine

data = np.loadtxt("./ex1data1.txt")
data = np.matrix(data)

x = data[:, 0]
y = data[:, 1]
m, n = np.shape(x)
x = np.hstack((np.ones([m, 1]), x))  # 合并矩阵

theta = np.zeros([n + 1, 1])
num_iteration = 2000  # 学习部署
alpha = 0.01  # 学习速率


def costFunction(x, y, theta):

    grads = np.zeros(np.shape(theta))
    m, n = np.shape(x)
    temp = np.dot(x, theta) - y
    cost = np.sum(np.multiply(temp, temp)) / (2 * m)  # 代价方程
    print("cost = ", cost)

    for i in range(len(theta)):
        grads[i] = np.sum(np.multiply(temp, x[:, i])) / m  # 计算梯度

    return grads


def gradientDescent(grads, alpha, theta):
    theta = theta - alpha * grads  # 梯度下降

    return theta


for i in range(num_iteration):  # 学习
    grads = costFunction(x, y, theta)
    theta = gradientDescent(grads, alpha, theta)


print("参数矩阵：")
print(theta)


plotDataAndDrawLine(x, y, theta)  # 画图
