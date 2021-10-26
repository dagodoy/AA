import time
from matplotlib import colors
import numpy as np
import scipy.integrate
import scipy.optimize as opt
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
from mpl_toolkits.mplot3d import Axes3D

#-----------------------------------------------------------------------

def carga_csv(file_name):

    valores = read_csv(file_name, header=None).to_numpy()
    return valores.astype(float)

#-----------------------------------------------------------------------
def regresion_logistica():
    datos = carga_csv('ex2data1.csv')
    X = datos[:, 0:2]
    Y = datos[:, 2]
    pos = np.where(Y==1)
    plt.scatter(X[pos,0], X[pos,1], marker='+', c='k')

    pos1 = np.where(Y==0)
    plt.scatter(X[pos1,0], X[pos1,1], marker='o', c='green')
    plt.savefig("grafica1.pdf")
    plt.clf()

    m=np.shape(X)[0]
    n=np.shape(X)[1]

    X = np.hstack([np.ones([m, 1]), X])

    theta = np.full(n+1, 0)
    print("Coste: ", cost(theta,X,Y))
    print("Gradiente: ", gradient(theta,X,Y))

    result = opt.fmin_tnc( func=cost , x0=theta , fprime=gradient , args =(X,Y))
    theta_opt = result[0]
    print("Coste Theta Op: ", cost(theta_opt,X,Y))

    pinta_frontera_recta(X,Y,theta_opt)

    print("Correctos: ", m-evaluacion(theta_opt,X,Y)) 


#-----------------------------------------------------------------------
def sigmoid(z):
    return 1/ (1+np.exp(-z))

#-----------------------------------------------------------------------
def cost(theta, X, Y):
    H = sigmoid(np.matmul(X, theta))
    cost = (- 1 / (len(X))) * (np.dot(Y, np.log(H)) + np.dot((1 - Y), np.log(1 - H)))
    return cost

#-----------------------------------------------------------------------

def gradient(theta, XX, Y):
    H = sigmoid( np.matmul(XX,theta))
    grad =(1/len(Y))* np.matmul(XX.T, H - Y)
    return grad

#-----------------------------------------------------------------------
def pinta_frontera_recta(X, Y, theta):
    plt.figure()
    pos = np.where(Y==1)
    plt.scatter(X[pos,1], X[pos,2], marker='+', c='k')

    pos1 = np.where(Y==0)
    plt.scatter(X[pos1,1], X[pos1,2], marker='o', c='green')

    x1_min, x1_max = X[:, 1].min(), X[:, 1].max()
    x2_min, x2_max = X[:, 2].min(), X[:, 2].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

    h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(theta))
    h = h.reshape(xx1.shape)

 # el cuarto parámetro es el valor de z cuya frontera se
 # quiere pintar
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
    plt.savefig("frontera.pdf")
    plt.close()

#-----------------------------------------------------------------------

def evaluacion(theta, X, Y):
    correctos = np.sum((np.dot(X, theta)>=0.5)==Y)
    return correctos
#-----------------------------------------------------------------------

regresion_logistica()
