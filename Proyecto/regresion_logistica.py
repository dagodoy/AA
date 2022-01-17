import numpy as np
import scipy.optimize as opt

#-----------------------------------------------------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

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

def coste_reg(theta, X, Y, lambd):
    H = sigmoid(np.matmul(X, theta))
    coste = cost(theta, X, Y) + lambd/(2*len(X)) * np.sum(theta**2)
    return coste

#-----------------------------------------------------------------------

def gradiente_reg(theta, XX, Y, lambd):
    H = sigmoid( np.matmul(XX,theta))
    grad =gradient(theta, XX, Y) + lambd/(len(XX)) * theta
    return grad

#-----------------------------------------------------------------------

def evaluacion(theta, X, Y):
    correctos = np.sum((sigmoid(np.dot(X, theta))>=0.5)==Y)
    return correctos

#-----------------------------------------------------------------------

def regresion_logistica_reg(X, Y, Xval, Yval):

    #como tenemos un monton de datos no hace falta hacer esto: X = mapfeature(X)

    m=np.shape(X)[0]
    n=np.shape(X)[1]

    X_ones = np.hstack([np.ones([m, 1]), X])

    mval = np.shape(Xval)[0]
    Xval_ones = np.hstack([np.ones([mval, 1]), Xval])

    max_correctos = 0
    max_lambda = 0

    #hasta 14880000000000000000000000 el parametro de regularizacion no tiene ninguna influencia sobre la precision del modelo, y cuando la tiene es a peor

    for lambd in np.arange(0, 0.0001, 0.00001):
        theta = np.full(n+1, 0)
        result = opt.fmin_tnc( func=coste_reg , x0=theta , fprime=gradiente_reg , args =(X_ones,Y, lambd), messages= 0)
        theta_opt = result[0]
        correctos = evaluacion(theta_opt,Xval_ones,Yval)
        if (correctos > max_correctos):
            max_correctos = correctos
            max_lambda = lambd


    return max_correctos/mval, max_lambda

#-----------------------------------------------------------------------