import time
from matplotlib import colors
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
from mpl_toolkits.mplot3d import Axes3D

#-----------------------------------------------------------------------

def carga_csv(file_name):

    valores = read_csv(file_name, header=None).to_numpy()
    return valores.astype(float)

#-----------------------------------------------------------------------

def regresion_unica():
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
    plt.clf()
    return [theta_0, theta_1]

#-----------------------------------------------------------------------

def coste(X,Y,theta):
    aux = 0
    m = len(X)
    for i in range(m):
        aux += ((theta[0] + theta[1] * X[i])-Y[i])**2
    return aux/(2*m)

#-----------------------------------------------------------------------

def make_data(t0_range, t1_range, X, Y):

    step = 0.1
    Theta0 = np.arange(t0_range[0], t0_range[1], step)
    Theta1 = np.arange(t1_range[0], t1_range[1], step)
    Theta0, Theta1 = np.meshgrid(Theta0, Theta1)

    Coste = np.empty_like(Theta0)
    for ix, iy in np.ndindex(Theta0.shape):
        Coste[ix, iy] = coste(X, Y, [Theta0[ix, iy], Theta1[ix, iy]])

    return [Theta0, Theta1, Coste]

#-----------------------------------------------------------------------

def dibuja_contorno(puntito, c):
    plt.plot(puntito[0],puntito[1], "x", color='red')
    plt.contour(c[0],c[1],c[2], np.logspace(-2, 3, 20), cmap='rainbow')

    plt.savefig("contorno.png") 
    plt.clf()

#-----------------------------------------------------------------------

def dibuja_superficie(c):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(c[0],c[1],c[2], cmap='rainbow',linewidth=0, antialiased=False)

    plt.savefig("superficie.png")   
    plt.clf()

#-----------------------------------------------------------------------

def graficas1():
    datos = carga_csv('ex1data1.csv')
    X = datos[:, 0]
    Y = datos[:, 1]

    puntito=regresion_unica()
    c = make_data([-10,10],[-1,4],X,Y)

    dibuja_contorno(puntito, c)
    dibuja_superficie(c)

#---------------------------------------------------------------------------------------------------------------------------------

def regresion_varias():
    datos = carga_csv('ex1data2.csv')

    X = datos[:, :-1] 
    Y = datos[:, -1]

    X_norm, x_mu, x_sigma = normalizar_mat(X)
    Y_norm, y_mu, y_simga = normalizar_mat(Y)

    m = np.shape(X)[0]
    n = np.shape(X)[1]

    # añadimos una columna de 1's a la X
    X_norm = np.hstack([np.ones([m, 1]), X])

    theta = np.full(n+1, 0)
    alpha = 0.01
    #Thetas, costes = descenso_gradiente(X, Y, alpha)

#-----------------------------------------------------------------------

def coste(X, Y, Theta):
    H = np.dot(X, Theta)
    Aux = (H - Y) ** 2
    return Aux.sum() / (2 * len(X))

#-----------------------------------------------------------------------

def normalizar_mat(X):
    mu = np.mean(X, 0)
    sigma = np.std(X, 0)
    X_norm = (X-mu)/sigma
    return X_norm, mu, sigma

#---------------------------------------------------------------------------------------------------------------------------------

#graficas1()
regresion_varias()