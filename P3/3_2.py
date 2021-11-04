
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
    theta1, theta2= loadRed()
    m = np.shape(X)[0]

    a1=np.hstack([np.ones([m, 1]), X])
    z2=np.dot(a1, theta1.T)
    a2=np.hstack([np.ones([m, 1]), sigmoid(z2)])
    z3=np.dot(a2, theta2.T)
    a3=sigmoid(z3)

    maxChance = a3.argmax(axis= 1)
    correctos = np.sum(maxChance == (y % 10))
    return correctos/m * 100

print (propagacion())
#-----------------------------------------------------------------------

# def oneVsAll(X, y, num_etiquetas, reg):
#     m = np.shape(X)[0]
#     X1s = np.hstack([np.ones([m, 1]), X])
#     theta_opt = np.zeros([num_etiquetas, X1s.shape[1]])

#     for i in range(num_etiquetas):
#         theta = np.zeros(X1s.shape[1]) 
#         if (i == 0):
#             result = opt.fmin_tnc( func=coste_reg , x0=theta , fprime=gradiente_reg , args =(X1s,(y == 10) *1, reg))
#         else:
#             result = opt.fmin_tnc( func=coste_reg , x0=theta , fprime=gradiente_reg , args =(X1s,(y == i) *1, reg))
#         theta_opt[i] = result[0]
#     return theta_opt

# #-----------------------------------------------------------------------

# def evaluacion(theta_opt, X, y):
#     m = np.shape(X)[0]
#     X1s = np.hstack([np.ones([m, 1]), X])

#     chances = np.zeros([np.shape(theta_opt)[0], np.shape(X1s)[0]])
#     for i in range(np.shape(theta_opt)[0]):
#         a = sigmoid(np.matmul(X1s,theta_opt[i]))
#         chances[i, :] = a


    
# #-----------------------------------------------------------------------

# def regresion_multi():
#     X, y = load()
#     y = y[:, 0]
#     theta_opt = oneVsAll(X, y, 10, 0.1)
#     print("Correctos: ", evaluacion(theta_opt, X, y))

#-----------------------------------------------------------------------
