
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt



#-----------------------------------------------------------------------

def load():
    data = loadmat('ex4data1.mat')
    y=data['y'].ravel()
    X=data['X']

    m = len(y)
    input_size = X.shape[1]
    num_labels = 10

    y = (y-1)
    y_onehot = np.zeros((m,num_labels))
    for i in range(m):
        y_onehot[i][y[i]] = 1

    return X, y_onehot

#-----------------------------------------------------------------------

def loadRed():
    weights = loadmat('ex4weights.mat')
    theta1, theta2 = weights['Theta1'],weights['Theta2']

    return theta1, theta2

#-----------------------------------------------------------------------

def sigmoid(z):
    return 1/ (1+np.exp(-z))

#-----------------------------------------------------------------------
def coste_red(theta, X, Y):
    _, _, H = propagacion(X,theta[0],theta[1])
    cost = (-1 / (len(X))) * np.sum((Y * np.log(H)) + (1 - Y) * np.log(1 - H + 1e-6))
    return cost

#-----------------------------------------------------------------------

def coste_red_reg(theta, X, Y, lambd):
    a = lambd/(2*(len(X))) * (np.sum(theta[0]**2) + np.sum(theta[1]**2))
    return coste_red(theta, X, Y) + a

#-----------------------------------------------------------------------

def gradiente_red(theta, XX, Y, lambd):
    H = sigmoid( np.matmul(XX,theta))
    grad =(1/len(Y))* np.matmul(XX.T, H - Y) + lambd/(len(XX)) * theta
    return grad

#-----------------------------------------------------------------------

def propagacion(X, Theta1, Theta2):
    
    m = X.shape[0]

    a1=np.hstack([np.ones([m, 1]), X])
    z2=np.dot(a1, Theta1.T)
    a2=np.hstack([np.ones([m, 1]), sigmoid(z2)])
    z3=np.dot(a2, Theta2.T)
    H=sigmoid(z3)

    return a1, a2, H

#-----------------------------------------------------------------------

def red_1():
    X, y = load()
    theta1, theta2 = loadRed()
    print ( coste_red_reg(np.array([theta1, theta2]), X, y, 1) )
    

red_1()
#-----------------------------------------------------------------------
