import time
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv


def carga_csv(file_name):

    valores = read_csv(file_name, header=None).to_numpy()
    return valores.astype(float)


def regresion():
    datos = carga_csv('ex1data1.csv')
    X = datos[:, 0]
    Y = datos[:, 1]
    m = len(X)
    alpha = 0.01
    theta_0 = theta_1 = 0
    for _ in range(1500):
        sum_0 = sum_1 = 0
        for i in range(m):
            sum_0 += (theta_0 + theta_1 * X[i]) - Y[i]
            sum_1 += ((theta_0 + theta_1 * X[i]) - Y[i]) * X[i]
        theta_0 = theta_0 - (alpha / m) * sum_0
        theta_1 = theta_1 - (alpha / m) * sum_1
    plt.plot(X, Y, "x")
    min_x = min(X)
    max_x = max(X)
    min_y = theta_0 + theta_1 * min_x
    max_y = theta_0 + theta_1 * max_x
    plt.plot([min_x, max_x], [min_y, max_y])
    plt.savefig("resultado.pdf")

def coste(X,Y,theta):
    aux = 0
    m = len(X)
    for i in range(m):
        aux += ((theta[0] + theta[1] * X[i])-Y[i])**2
    return aux/(2*m)

def make_data(t0_range, t1_range, X, Y):

    step = 0.1
    Theta0 = np.arange(t0_range[0], t0_range[1], step)
    Theta1 = np.arange(t1_range[0], t1_range[1], step)
    Theta0, Theta1 = np.meshgrid(Theta0, Theta1)

    Coste = np.empty_like(Theta0)
    for ix, iy in np.ndindex(Theta0.shape):
        Coste[ix, iy] = coste(X, Y, [Theta0[ix, iy], Theta1[ix, iy]])

    # puntito que ya tal
    plt.contour(Theta0, Theta1, Coste, np.logspace(-2, 3, 20), colors='blue') # cmap='rainbow'
    plt.savefig("contorno.png")    

    return [Theta0, Theta1, Coste]

datos = carga_csv('ex1data1.csv')
X = datos[:, 0]
Y = datos[:, 1]
t0=[-10,10]
t1=[-1,4]

make_data(t0,t1,X,Y)