
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt



#-----------------------------------------------------------------------

def load():
    data = loadmat('ex3data1.mat')
    y=data['y']
    X=data['X']

    sample = np.random.choice(X.shape[0], 10)
    plt.imshow(X[sample, :].reshape(-1,20).T)
    plt.axis('off')

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

def oneVsAll(X, y, num_etiquetas, reg):
    for i in range(num_etiquetas):

load()
# def regresion_logistica():
#     datos = carga_csv('ex2data1.csv')
#     X = datos[:, 0:2]
#     Y = datos[:, 2]
#     pos = np.where(Y==1)
#     plt.scatter(X[pos,0], X[pos,1], marker='+', c='k')

#     pos1 = np.where(Y==0)
#     plt.scatter(X[pos1,0], X[pos1,1], marker='o', c='green')
#     plt.savefig("grafica1.png")
#     plt.clf()

#     m=np.shape(X)[0]
#     n=np.shape(X)[1]

#     X = np.hstack([np.ones([m, 1]), X])

#     theta = np.full(n+1, 0)
#     print("Coste inicial: ", cost(theta,X,Y))
#     print("Gradiente: ", gradient(theta,X,Y))

#     result = opt.fmin_tnc( func=cost , x0=theta , fprime=gradient , args =(X,Y))
#     theta_opt = result[0]
#     print("Coste Theta Optimizada: ", cost(theta_opt,X,Y))

#     pinta_frontera_recta(X,Y,theta_opt)

#     print("Correctos: ", evaluacion(theta_opt,X,Y) , " de ", m) 


# #-----------------------------------------------------------------------
# def sigmoid(z):
#     return 1/ (1+np.exp(-z))

# #-----------------------------------------------------------------------
# def cost(theta, X, Y):
#     H = sigmoid(np.matmul(X, theta))
#     cost = (- 1 / (len(X))) * (np.dot(Y, np.log(H)) + np.dot((1 - Y), np.log(1 - H)))
#     return cost

# #-----------------------------------------------------------------------

# def gradient(theta, XX, Y):
#     H = sigmoid( np.matmul(XX,theta))
#     grad =(1/len(Y))* np.matmul(XX.T, H - Y)
#     return grad

# #-----------------------------------------------------------------------
# def pinta_frontera_recta(X, Y, theta):
#     plt.figure()
#     pos = np.where(Y==1)
#     plt.scatter(X[pos,1], X[pos,2], marker='+', c='k')

#     pos1 = np.where(Y==0)
#     plt.scatter(X[pos1,1], X[pos1,2], marker='o', c='green')

#     x1_min, x1_max = X[:, 1].min(), X[:, 1].max()
#     x2_min, x2_max = X[:, 2].min(), X[:, 2].max()

#     xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

#     h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(theta))
#     h = h.reshape(xx1.shape)

#  # el cuarto parámetro es el valor de z cuya frontera se
#  # quiere pintar
#     plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
#     plt.savefig("frontera.png")
#     plt.close()

# #-----------------------------------------------------------------------

# def evaluacion(theta, X, Y):
#     correctos = np.sum((sigmoid(np.dot(X, theta))>=0.5)==Y)
#     return correctos

# #-----------------------------------------------------------------------

# def mapfeature(X):
#     poly = PolynomialFeatures(6)
#     return poly.fit_transform(X)

# #-----------------------------------------------------------------------

# def coste_reg(theta, X, Y, lambd):
#     H = sigmoid(np.matmul(X, theta))
#     cost = (- 1 / (len(X))) * (np.dot(Y, np.log(H)) + np.dot((1 - Y), np.log(1 - H))) + lambd/(2*len(X)) * np.sum(theta**2)
#     return cost

# #-----------------------------------------------------------------------

# def gradiente_reg(theta, XX, Y, lambd):
#     H = sigmoid( np.matmul(XX,theta))
#     grad =(1/len(Y))* np.matmul(XX.T, H - Y) + lambd/(len(XX)) * theta
#     return grad

# #-----------------------------------------------------------------------

# def plot_decisionboundary(X, Y, theta, poly):
#     plt.figure()

#     pos = np.where(Y==1)
#     plt.scatter(X[pos,1], X[pos,2], marker='+', c='k')

#     pos1 = np.where(Y==0)
#     plt.scatter(X[pos1,1], X[pos1,2], marker='o', c='green')
 
#     x1_min, x1_max = X[:, 1].min(), X[:, 1].max()
#     x2_min, x2_max = X[:, 2].min(), X[:, 2].max()
#     xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
#                            np.linspace(x2_min, x2_max))
 
#     h = sigmoid(poly.fit_transform(np.c_[xx1.ravel(),
#                                          xx2.ravel()]).dot(theta))
#     h = h.reshape(xx1.shape)
 
#     plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')
#     plt.savefig("lambda 100.png")
#     plt.close()

# #-----------------------------------------------------------------------

# def regresion_logistica_reg():

#     datos = carga_csv('ex2data2.csv')
#     X = datos[:, 0:2]
#     Y = datos[:, 2]
    
#     pos = np.where(Y==1)
#     plt.scatter(X[pos,0], X[pos,1], marker='+', c='k')

#     pos1 = np.where(Y==0)
#     plt.scatter(X[pos1,0], X[pos1,1], marker='o', c='green')
#     plt.savefig("grafica2.png")
#     plt.clf()

#     X = mapfeature(X)

#     m=np.shape(X)[0]
#     n=np.shape(X)[1]

#     theta = np.full(n, 0)
#     print("Coste inicial regularizado: ", coste_reg(theta,X,Y,1))

#     result = opt.fmin_tnc( func=coste_reg , x0=theta , fprime=gradiente_reg , args =(X,Y, 1))
#     theta_opt = result[0]

#     plot_decisionboundary(X, Y, theta_opt, PolynomialFeatures(6))

#-----------------------------------------------------------------------

