
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

def gradiente_lineal(Theta, X, Y, lambd):
    H = np.dot(X, Theta)
    gradiente = np.dot((H-Y),X)/len(X)
    gradiente[1:] = gradiente[1:] + (lambd * Theta/len(X))[1:]
    return gradiente
    
#-----------------------------------------------------------------------

def lineal(Theta, X, Y, lambd):
    return coste_lineal(X, Y, Theta, lambd), gradiente_lineal(Theta, X, Y, lambd)
    
#-----------------------------------------------------------------------

X, y, Xval, yval, Xtest, ytest = load()
m = np.shape(X)[0]
X = np.hstack([np.ones([m, 1]), X])
y = y.ravel()

theta = np.array([1, 1])

res = opt.minimize(fun=lineal,x0= theta, args= (X, y, 1), jac = True, method = 'TNC')

plt.plot(X[:, 1], y, "x")
min_x = min(X[:, 1])
max_x = max(X[:, 1])
min_y = res.x[0] + res.x[1] * min_x
max_y = res.x[0] + res.x[1] * max_x
plt.plot([min_x, max_x], [min_y, max_y])
plt.savefig("resultado.pdf")
plt.clf()