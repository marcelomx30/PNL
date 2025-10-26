"""Funções de Teste - Projeto Busca Linear - Grupo 04"""
import numpy as np

# Funções e gradientes
def f1(x):
    return (x[0] - 2)**4 + (x[0] - 2*x[1])**2

def grad_f1(x):
    return np.array([4*(x[0]-2)**3 + 2*(x[0]-2*x[1]), -4*(x[0]-2*x[1])])

def f2(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

def grad_f2(x):
    return np.array([-400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0]), 200*(x[1]-x[0]**2)])

def f3(x):
    t1 = (1.5 - x[0]*(1-x[1]))**2
    t2 = (2.25 - x[0]*(1-x[1]**2))**2
    t3 = (2.625 - x[0]*(1-x[1]**3))**2
    return t1 + t2 + t3

def grad_f3(x):
    t1 = 1.5 - x[0]*(1-x[1])
    t2 = 2.25 - x[0]*(1-x[1]**2)
    t3 = 2.625 - x[0]*(1-x[1]**3)
    df1 = -2*t1*(1-x[1]) - 2*t2*(1-x[1]**2) - 2*t3*(1-x[1]**3)
    df2 = 2*t1*x[0] + 4*t2*x[0]*x[1] + 6*t3*x[0]*x[1]**2
    return np.array([df1, df2])

def f4(x):
    s = x[0] + x[1]
    d = x[0] - x[1]
    q = (x[0]-2)**2 + x[1]**2 - 1
    return (4*s)**2 + (4*s + d*q)**2

def grad_f4(x):
    s = x[0] + x[1]
    d = x[0] - x[1]
    q = (x[0]-2)**2 + x[1]**2 - 1
    t1 = 4*s
    t2 = 4*s + d*q
    df1 = 8*t1 + 2*t2*(4 + q + d*2*(x[0]-2))
    df2 = 8*t1 + 2*t2*(4 - q + d*2*x[1])
    return np.array([df1, df2])

def f5(x):
    avg = (x[0] + x[1])/2
    return 100*(x[2]-avg)**2 + (1-x[0])**2 + (1-x[1])**2

def grad_f5(x):
    avg = (x[0] + x[1])/2
    return np.array([
        -100*(x[2]-avg) - 2*(1-x[0]),
        -100*(x[2]-avg) - 2*(1-x[1]),
        200*(x[2]-avg)
    ])

def f6(x):
    return (100*(x[1]-x[0]**2)**2 + (1-x[0])**2 + 90*(x[3]-x[2]**2)**2 + 
            (1-x[2])**2 + 10.1*((x[1]-1)**2 + (x[3]-1)**2) + 19.8*(x[1]-1)*(x[3]-1))

def grad_f6(x):
    return np.array([
        -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0]),
        200*(x[1]-x[0]**2) + 20.2*(x[1]-1) + 19.8*(x[3]-1),
        -360*x[2]*(x[3]-x[2]**2) - 2*(1-x[2]),
        180*(x[3]-x[2]**2) + 20.2*(x[3]-1) + 19.8*(x[1]-1)
    ])

FUNCOES = {
    'f1': (f1, grad_f1), 'f2': (f2, grad_f2), 'f3': (f3, grad_f3),
    'f4': (f4, grad_f4), 'f5': (f5, grad_f5), 'f6': (f6, grad_f6)
}

CASOS_TESTE = {
    'f1': [{'x': np.array([0., 3.]), 'nome': 'caso_i'},
           {'x': np.array([-1., -1.]), 'nome': 'caso_ii'}],
    'f2': [{'x': np.array([-1.9, 2.]), 'nome': 'caso_i'},
           {'x': np.array([1.2, 1.]), 'nome': 'caso_ii'}],
    'f3': [{'x': np.array([0., 0.]), 'nome': 'caso_i'},
           {'x': np.array([5., 5.]), 'nome': 'caso_ii'}],
    'f4': [{'x': np.array([2., 0.]), 'nome': 'caso_i'},
           {'x': np.array([-2., 2.]), 'nome': 'caso_ii'}],
    'f5': [{'x': np.array([0., 0., 0.]), 'nome': 'caso_i'},
           {'x': np.array([-1.2, 2., 0.]), 'nome': 'caso_ii'}],
    'f6': [{'x': np.array([0., 0., 0., 0.]), 'nome': 'caso_i'},
           {'x': np.array([-3., -1., -3., -1.]), 'nome': 'caso_ii'}]
}
