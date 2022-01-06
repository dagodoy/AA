
from math import exp
from warnings import resetwarnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.svm import SVC
import process_email
from get_vocab_dict import getVocabDict
import codecs


#-----------------------------------------------------------------------

def loadData(file):
    
    data = loadmat(file)
    y=data['y']
    X=data['X']
    
    return X,y

#-----------------------------------------------------------------------

def loadData3():
    data = loadmat('ex6data3.mat')
    y=data['y']
    X=data['X']
    
    Xval=data['Xval']
    yval=data['yval']
    
    return X,y,Xval,yval
#-----------------------------------------------------------------------

def visualize_boundary(X, y, svm, file_name):
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    x1, x2 = np.meshgrid(x1, x2)
    yp = svm.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)
    pos = (y == 1).ravel()
    neg = (y == 0).ravel()
    plt.figure()
    plt.scatter(X[pos, 0], X[pos, 1], color='black', marker='+')
    plt.scatter(X[neg, 0], X[neg, 1], color='yellow', edgecolors='black', marker='o')
    plt.contour(x1, x2, yp)
    plt.savefig(file_name)
    plt.close()

#-----------------------------------------------------------------------

def calculaScore(svm, Xval, yval):
    yp = svm.predict(Xval).reshape(yval.shape)
    return sum(yp == yval)


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

def parte1():
    X, y = loadData('ex6data1.mat')
    svm = SVC(kernel= 'linear' , C=1.0)
    svm.fit(X, y)
    visualize_boundary(X, y,svm ,'data1_1.png')

    X, y = loadData('ex6data2.mat')
    C = 1
    sigma = 0.1
    svm = SVC(kernel= 'rbf' , C=C, gamma = 1/(2*sigma**2))
    svm.fit(X, y)
    visualize_boundary(X, y,svm ,'data1_2.png')

    X, y, Xval, yval = loadData3()
    svm = eleccionParams(X, y, Xval, yval)
    visualize_boundary(X, y, svm ,'data1_3.png')

#-----------------------------------------------------------------------

def cargaEmails(directorio, nFiles):
    vocab = getVocabDict()
    emails = np.zeros((nFiles, len(vocab)))
    for i in range(1, nFiles+1):
        email_contents = codecs.open('{0}/{1:04d}.txt'.format(directorio, i), 'r', encoding='utf-8', errors='ignore').read()
        words = process_email.email2TokenList(email_contents)
        vec = np.zeros(len(vocab))
        for w in words:
            if w in vocab:
                vec[vocab[w]-1] = 1
        emails[i-1] = vec
    return emails

#-----------------------------------------------------------------------
def parte2():
    spam = cargaEmails("spam", 500)
    easy_ham = cargaEmails("easy_ham", 2551)
    hard_ham =cargaEmails("hard_ham", 250)
    return spam, easy_ham, hard_ham



#-----------------------------------------------------------------------

parte2()