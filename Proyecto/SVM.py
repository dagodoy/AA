import numpy as np
from sklearn.svm import SVC

#-----------------------------------------------------------------------

def calculaScore(svm, Xval, yval):
    yp = svm.predict(Xval).reshape(yval.shape)
    return sum(yp == yval)

#-----------------------------------------------------------------------

def eleccionParams(X, y, Xval, yval):
    C_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigma_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    scores = np.zeros((len(C_vec), len(sigma_vec)))
    
    for i in range(len(C_vec)):
        for j in range(len(sigma_vec)):
            svm = SVC(kernel= 'rbf' , C=C_vec[i], gamma = 1/(2*sigma_vec[j]**2))
            svm.fit(X, y)
            scores[i][j] = calculaScore(svm, Xval, yval)

    index = np.unravel_index(np.argmax(scores), scores.shape)

    svm = SVC(kernel= 'rbf' , C=C_vec[index[0]], gamma = 1/(2*sigma_vec[index[1]]**2))
    svm.fit(X, y)
    return svm
    
#-----------------------------------------------------------------------

def svm_proyecto( X, y, Xval, yval):
    
    svm = eleccionParams(X, y, Xval, yval)
    return calculaScore(svm,Xval, yval)/np.shape(Xval)[0]

#-----------------------------------------------------------------------