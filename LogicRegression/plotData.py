# @Author: guchenghao
# @Date:   2017-04-25T20:13:22+08:00
# @Email:  1754926878@qq.com
# @Last modified by:   guchenghao
# @Last modified time: 2017-05-19T17:09:02+08:00


import matplotlib.pyplot as plt
import numpy as np


def plotData(x, y):
    numEx, numFe = np.shape(x)

    for i in range(numEx):
        if int(y[i, 0]) == 0:
            plt.plot(x[i, 0], x[i, 1], 'or')
        elif int(y[i, 0]) == 1:
            plt.plot(x[i, 0], x[i, 1], 'ob')
