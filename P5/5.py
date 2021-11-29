
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt


#-----------------------------------------------------------------------

def load():
    data = loadmat('ex5data1.mat')
    y=data['y']
    X=data['X']
    
    Xval=data['Xval']
    yval=data['yval']

    Xtest=data['Xtest']
    ytest=data['ytest']
    
    return X,y,Xval,yval,Xtest,ytest
#-----------------------------------------------------------------------

def coste_lineal(X, Y, Theta, lambd):
    H = np.dot(X, Theta)
    return (np.sum((H - Y) ** 2) + np.sum(lambd * (Theta[1:] ** 2)))/(2*len(X))
#-----------------------------------------------------------------------

X, y, Xval, yval, Xtest, ytest = load()
m = np.shape(X)[0]
X = np.hstack([np.ones([m, 1]), X])
y = y.ravel()

theta = np.array([1, 1])


print (coste_lineal(X, y, theta, 1))
