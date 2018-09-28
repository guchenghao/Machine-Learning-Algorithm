from plotData import plotData
import matplotlib.pyplot as plt


def plotDecisionBoundary(x, y, theta):

    plotData(x[:, 1:3], y)
    plot_min_x = min(x[:, 2])[0, 0] - 2
    plot_max_x = max(x[:, 2])[0, 0] + 2

    plot_min_y = (-1.0 / theta[2]) * (theta[1] * plot_min_x + theta[0])
    plot_max_y = (-1.0 / theta[2]) * (theta[1] * plot_max_x + theta[0])

    plt.plot([plot_min_x, plot_max_x], [plot_min_y, plot_max_y])
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
