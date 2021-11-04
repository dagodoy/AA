
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt



#-----------------------------------------------------------------------

def load():
    data = loadmat('ex3data1.mat')
    y=data['y']
    X=data['X']

    return X, y

#-----------------------------------------------------------------------

def loadRed():
    weights = loadmat('ex3weights.mat')
    theta1, theta2 = weights['Theta1'],weights['Theta2']

    return theta1, theta2

#-----------------------------------------------------------------------

def sigmoid(z):
    return 1/ (1+np.exp(-z))

#-----------------------------------------------------------------------
def coste_reg(theta, X, Y, lambd):
    H = sigmoid(np.matmul(X, theta))
    cost = (- 1 / (len(X))) * (np.dot(Y, np.log(H)) + np.dot((1 - Y), np.log(1 - H + 1e-6))) + lambd/(2*len(X)) * np.sum(theta**2)
    return cost

#-----------------------------------------------------------------------

def gradiente_reg(theta, XX, Y, lambd):
    H = sigmoid( np.matmul(XX,theta))
    grad =(1/len(Y))* np.matmul(XX.T, H - Y) + lambd/(len(XX)) * theta
    return grad

#-----------------------------------------------------------------------

def propagacion():
    X, y = load()
    y = y.ravel()
    theta1, theta2= loadRed()
    m = np.shape(X)[0]

    a1=np.hstack([np.ones([m, 1]), X])
    z2=np.dot(a1, theta1.T)
    a2=np.hstack([np.ones([m, 1]), sigmoid(z2)])
    z3=np.dot(a2, theta2.T)
    a3=sigmoid(z3)

    maxChance = a3.argmax(axis= 1)
    correctos = np.sum(maxChance+1 == y)
    return correctos/m * 100

print ("Correctos: ", propagacion())
#-----------------------------------------------------------------------
