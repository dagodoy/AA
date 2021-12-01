
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
    return (np.sum((H - Y.T) ** 2) + np.sum(lambd * (Theta[1:] ** 2)))/(2*len(X))
    
#-----------------------------------------------------------------------

def gradiente_lineal(Theta, X, Y, lambd):
    H = np.dot(X, Theta)
    gradiente = np.dot((H-Y),X)/len(X)
    gradiente[1:] = gradiente[1:] + (lambd * Theta/len(X))[1:]
    return gradiente
    
#-----------------------------------------------------------------------

def lineal(Theta, X, Y, lambd):
    return coste_lineal(X, Y, Theta, lambd), gradiente_lineal(Theta, X, Y, lambd)
    
#-----------------------------------------------------------------------

def error(X,y,reg,Xval, Yval):
    m = np.shape(X)[0]
    mV = np.shape(Xval)[0]
    errorV = np.zeros([m])
    errorE = np.zeros([m])

    Xval = np.hstack([np.ones([mV, 1]), Xval])

    for i in range(1,m+1):
        theta = np.zeros(np.shape(X)[1])
        res = opt.minimize(fun=lineal,x0= theta, args= (X[0:i], y[0:i], reg), jac = True, method = 'TNC')
        errorV[i-1] = coste_lineal(Xval,Yval,res.x,0)
        errorE[i-1] = coste_lineal(X[0:i],y[0:i],res.x,0)

    return errorE, errorV


#-----------------------------------------------------------------------

def pintaLineal(X, y):

    theta = np.array([1, 1])

    res = opt.minimize(fun=lineal,x0= theta, args= (X, y, 1), jac = True, method = 'TNC')

    plt.plot(X[:, 1], y, "x")
    min_x = min(X[:, 1])
    max_x = max(X[:, 1])
    min_y = res.x[0] + res.x[1] * min_x
    max_y = res.x[0] + res.x[1] * max_x
    plt.plot([min_x, max_x], [min_y, max_y])
    plt.savefig("resultado.png")
    plt.clf()

#-----------------------------------------------------------------------

def pintaError(X, y, Xval, yval):

    errorE, errorV = error(X,y,0,Xval, yval)


    plt.plot(np.linspace(1,11,12,dtype=int),errorE, label="Train")
    plt.plot(np.linspace(1,11,12,dtype=int),errorV, label="Cross Validation")
    plt.legend()
    plt.savefig("curvas.png")
    plt.clf()

def main():
    X, y, Xval, yval, Xtest, ytest = load()
    m = np.shape(X)[0]
    X = np.hstack([np.ones([m, 1]), X])
    y = y.ravel()

    pintaLineal(X, y)
    pintaError(X, y, Xval, yval)

#-----------------------------------------------------------------------
main()