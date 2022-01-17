
from pandas.io.parsers import read_csv

from regresion_logistica import regresion_logistica_reg 



#-----------------------------------------------------------------------

def carga_csv(file_name):

    valores = read_csv(file_name, header=0).to_numpy()
    return valores.astype(float)

#-----------------------------------------------------------------------

data = carga_csv('data.csv')
size = (data.shape[0]/2).__int__()

X = data[0:size, 1:]
y = data[0:size, 0]

Xval = data[size:, 1:]
yval = data[size:, 0]

lambd = 1

print(regresion_logistica_reg(X, y, Xval, yval))
