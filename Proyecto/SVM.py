import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------

def calculaScore(svm, Xval, yval):
    yp = svm.predict(Xval).reshape(yval.shape)
    return sum(yp == yval)

#-----------------------------------------------------------------------

def eleccionParams(X, y, Xval, yval):
    C_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigma_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    scores = np.zeros((len(C_vec), len(sigma_vec)))

    plt.figure()
    for i in range(len(C_vec)):
        for j in range(len(sigma_vec)):
            svm = SVC(kernel= 'rbf' , C=C_vec[i], gamma = 1/(2*sigma_vec[j]**2))
            svm.fit(X, y)
            scores[i][j] = calculaScore(svm, Xval, yval)

    axes = plt.axes(projection= '3d')
    axes.contour(C_vec, sigma_vec, scores, cmap='rainbow',linewidth=0, antialiased=False)
    axes.set_xlabel('C')
    axes.set_ylabel('sigma')
    axes.set_zlabel('nº aciertos')
    plt.savefig("svm.png")
    index = np.unravel_index(np.argmax(scores), scores.shape)

    svm = SVC(kernel= 'rbf' , C=C_vec[index[0]], gamma = 1/(2*sigma_vec[index[1]]**2))
    svm.fit(X, y)
    return svm
    
#-----------------------------------------------------------------------

def svm_proyecto( X, y, Xval, yval):
    
    svm = eleccionParams(X, y, Xval, yval)
    return calculaScore(svm,Xval, yval)/np.shape(Xval)[0]

#-----------------------------------------------------------------------