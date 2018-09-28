import numpy as np
import plotDecisionBoundary as pDB


data = np.loadtxt("./ex2data.txt")  # 导入训练集(txt文件)
data = np.matrix(data)

x = data[:, 0:2]
m, n = np.shape(x)
x = np.hstack((np.ones([m, 1]), x))  # 矩阵合并
y = data[:, 2]

theta = np.zeros([n + 1, 1])  # theta矩阵

alpha = 0.03  # 初始化学习速率


def sigmoid(z):  # sigmoid函数
    return 1.0 / (1.0 + np.exp(-z))


def costFunction(x, y, theta):  # 代价函数

    hypo = sigmoid(x * theta)
    grads = np.zeros(np.shape(theta))  # 初始化

    a, b = np.shape(y)
    j = np.multiply(y, np.log(hypo)) + np.multiply(1.0 - y, np.log(1.0 - hypo))
    cost = np.sum(j) * (- 1.0 / a)  # 计算出代价值
    print("cost = ", cost)

    # 计算出每个feature的梯度值
    # ! 每次都需要根据最新的theta值，来更新梯度向量(列向量)
    for i in range(len(theta)):
        temp = np.multiply(hypo - y, x[:, i])
        grads[i] = (1.0 / a) * np.sum(temp)
        temp = 0

    return grads


def gradientDescent(grads, alpha, theta):  # 使用梯度下降算法

    theta = theta - alpha * grads

    return theta


for i in range(2500):  # 开始学习，2500步
    grads = costFunction(x, y, theta)  # 更新梯度矩阵，并打印当前的cost值
    theta = gradientDescent(grads, alpha, theta)


print("参数矩阵：")
print(theta)

pDB.plotDecisionBoundary(x, y, theta)  # 画出决策边界
