
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
#import checkNNGradients


#-----------------------------------------------------------------------

def load(X, y, XVal,Yval):
    # data = loadmat('ex4data1.mat')
    # y=data['y'].ravel()
    # X=data['X']

    m = len(y)
    input_size = X.shape[1]
    num_labels = 2

    y = (y-1)
    y_onehot = np.zeros((m,num_labels))
    for i in range(m):
        y_onehot[i][y[i].__int__()] = 1

    n= len(Yval)
    Yval = (Yval-1)
    yval_onehot = np.zeros((n,num_labels))
    for i in range(n):
        yval_onehot[i][Yval[i].__int__()] = 1

    return X, y_onehot, yval_onehot

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
    cost = (-1 / (len(X))) * np.sum((Y * np.log(H)) + (1 - Y) * np.log(1 - H + 1e-9))
    return cost

#-----------------------------------------------------------------------

def coste_red_reg(theta, X, Y, lambd):
    a = lambd/(2*(len(X))) * (np.sum(theta[0][1:]**2) + np.sum(theta[1][1:]**2))
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

def termino_reg(g, m, reg, theta):
    columna = g[0]
    g = g + (reg/m)*theta
    g[0] = columna
    return g

#-----------------------------------------------------------------------

def backprop ( params_rn , num_entradas , num_ocultas , num_etiquetas , X, y , reg):

    Theta1 = np.reshape ( params_rn [ : num_ocultas * ( num_entradas + 1 ) ] ,( num_ocultas , ( num_entradas + 1 )))
    Theta2 = np.reshape ( params_rn [ num_ocultas * ( num_entradas + 1 ) : ] ,( num_etiquetas , ( num_ocultas + 1 )))

    A1, A2, H = propagacion(X, Theta1, Theta2)
    m = X.shape[0]

    Delta1 = np.zeros_like(Theta1)
    Delta2 = np.zeros_like(Theta2)

    for t in range(m):
        a1t = A1[t, :] # (401,)
        a2t = A2[t, :] # (26,)
        ht = H[t, :] # (10,)
        yt = y[t] # (10,)
        d3t = ht - yt # (10,)
        d2t = np.dot(Theta2.T, d3t) * (a2t * (1 - a2t)) # (26,)
        Delta1 = Delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
        Delta2 = Delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])

    G1 = Delta1/m
    G2 = Delta2/m

    G1 = termino_reg(G1, m, reg, Theta1)
    G2 = termino_reg(G2, m, reg, Theta2)

    return coste_red_reg(np.array([Theta1, Theta2]), X, y, reg), np.concatenate([np.ravel(G1), np.ravel(G2)])        

#-----------------------------------------------------------------------

def red_1(X, y ,XVal, YVal):
    X, y, yval = load(X, y ,XVal, YVal)
    # theta1, theta2 = loadRed()
    # params_rn = np.concatenate([np.ravel(theta1), np.ravel(theta2)])
    num_entradas = 95
    num_ocultas = 1000
    num_etiquetas = 2

    # tupla = backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y,1 )

    #checkNNGradients.checkNNGradients(backprop, 1)

    ini = 0.12
    reg=1
    i=100

    pesos= np.random.uniform(-ini,ini,(num_entradas+1)*num_ocultas+(num_ocultas+1)*num_etiquetas)
    sol = opt.minimize(fun=backprop, x0=pesos, args=(num_entradas,num_ocultas, num_etiquetas,X, y ,reg), jac= True,method = 'TNC', options ={'maxiter' :i})

    theta1 = np.reshape ( sol.x [ : num_ocultas * ( num_entradas + 1 ) ] ,( num_ocultas , ( num_entradas + 1 )))
    theta2 = np.reshape ( sol.x [ num_ocultas * ( num_entradas + 1 ) : ] ,( num_etiquetas , ( num_ocultas + 1 )))
    
    m = np.shape(X)[0]

    A1, A2, H = propagacion(XVal, theta1, theta2)

    maxChance = H.argmax(axis= 1)
    res = yval.argmax(axis= 1)
    correctos = np.sum(maxChance == res)
    return correctos/m * 100
    

#-----------------------------------------------------------------------
