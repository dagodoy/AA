import time
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

def integra_mc(fun, a, b, num_puntos = 10000):
    dentro = 0
    max = np.amax(fun(np.linspace(a,b,num_puntos)))

    tic = time.process_time()

    for i in range(num_puntos):
        x = np.random.uniform(a, b)
        y = np.random.uniform(0, max)
        if (fun(x) < y):
            dentro = dentro + 1

    toc = time.process_time()
    sol = (dentro/num_puntos) * (b-a) * max
    return 1000 * (toc-tic)

def integra_mc_vec(fun, a, b, num_puntos = 10000):
    max = np.amax(fun(np.linspace(a,b,num_puntos)))

    tic = time.process_time()

    x = np.random.uniform(a, b, num_puntos)
    y = fun(x)
    yrand = np.random.uniform(0, max, num_puntos)
    
    toc = time.process_time()
    sol = (sum(yrand<y)/num_puntos) * (b-a) * max
    return 1000 * (toc-tic)

def compara_tiempos():
    sizes = np.linspace(10, 100000, 20, dtype=int)
    times_it = []
    times_vec = []
    for size in sizes:
        times_it += [integra_mc(funcion, 1, 10, size)]
        times_vec += [integra_mc_vec(funcion, 1, 10, size)]
    
    plt.figure()
    plt.scatter(sizes, times_it, c='red', label='iterativo')
    plt.scatter(sizes, times_vec, c='blue', label='array')
    plt.legend()
    plt.savefig('time.png')


def funcion(x):
    return 2*x

compara_tiempos()
# a = scipy.integrate.quad(funcion, 0, 100)
# b = integra_mc_vec(funcion, 0, 100, 1000000)
# print(a)
# print(b)

