
from pandas.io.parsers import read_csv

from regresion_logistica import regresion_logistica_reg 
from redes_neuronales import red_neuronal
from SVM import svm_proyecto
import numpy as np
import matplotlib.pyplot as plt


#-----------------------------------------------------------------------

def carga_csv(file_name):

    valores = read_csv(file_name, header=0).to_numpy()
    return valores.astype(float)

#-----------------------------------------------------------------------

def normalizar_mat(X):
    mu = np.mean(X, 0)
    sigma = np.std(X, 0)
    X_norm = (X-mu)/sigma
    X_norm[93] += X[93]
    return X_norm, mu, sigma
#-----------------------------------------------------------------------

data = carga_csv('data.csv')
data = np.delete(data, 94, 1)
size = (data.shape[0]/2).__int__()

X = data[0:size, 1:]
y = data[0:size, 0]

Xval = data[size:, 1:]
yval = data[size:, 0]

XvalNor, _, _ = normalizar_mat(Xval)
XNor, _, _ = normalizar_mat(X)

_reg , _ = (regresion_logistica_reg(XNor, y, XvalNor, yval))
_red, _ = (red_neuronal(XNor,y,XvalNor,yval))
_svm = (svm_proyecto(XNor,y,XvalNor,yval))
print(_reg)
print(_red)
print(_svm)

xBars = ['Reg. Logistica', 'Red Neuronal', 'SVM']
ancho = 0.7
fig, aux = plt.subplots(figsize=(8,7))
index = np.arange(len(xBars))
plt.bar(index, [float(_reg), float(_red), float(_svm)], ancho, color = '#7986CB')
plt.xticks(index, xBars, fontsize=9)
plt.ylim((0,1))
plt.savefig('Comparacion1.png')

xBars = ['Reg. Logistica', 'Red Neuronal', 'SVM']
ancho = 0.7
fig, aux = plt.subplots(figsize=(8,7))
index = np.arange(len(xBars))
plt.bar(index, [float(_reg), float(_red), float(_svm)], ancho, color = '#7986CB')
plt.xticks(index, xBars, fontsize=9)
plt.ylim((0.8,1))
plt.savefig('Comparacion2.png')

xBars = ['Reg. Logistica', 'Red Neuronal', 'SVM']
ancho = 0.7
fig, aux = plt.subplots(figsize=(8,7))
index = np.arange(len(xBars))
plt.bar(index, [float(_reg), float(_red), float(_svm)], ancho, color = '#7986CB')
plt.xticks(index, xBars, fontsize=9)
plt.ylim((0.925,1))
plt.savefig('Comparacion3.png')