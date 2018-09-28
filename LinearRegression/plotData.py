# @Author: guchenghao
# @Date:   2017-04-26T18:34:24+08:00
# @Email:  1754926878@qq.com
# @Last modified by:   guchenghao
# @Last modified time: 2017-05-21T21:55:54+08:00


import matplotlib.pyplot as plt
import numpy as np


def plotDataAndDrawLine(x, y, theta):

    numEx, numFe = np.shape(x)  # 画出所有样本
    for i in range(numEx):
        plt.plot(x[i, 1], y[i, 0], 'or')

    plot_min_x = min(x[:, 1])[0, 0] - 2
    plot_max_x = max(x[:, 1])[0, 0] + 2

    plot_min_y = theta[1] * plot_min_x + theta[0]
    plot_max_y = theta[1] * plot_max_x + theta[0]

    plt.plot([plot_min_x, plot_max_x], [plot_min_y, plot_max_y])  # 拟合出直线
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
