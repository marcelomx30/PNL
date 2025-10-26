"""Método da Seção Áurea - Grupo 04"""
import numpy as np
import time

TAU = (np.sqrt(5) - 1) / 2  # ~0.618

def secao_aurea(phi, a=0, b=1, tol=1e-5, max_iter=1000):
    """
    Método da Seção Áurea (sem derivada)
    
    Retorna: (alpha_otimo, f_otimo, iteracoes, tempo, convergiu)
    """
    inicio = time.time()
    
    x1 = a + (1 - TAU) * (b - a)
    x2 = a + TAU * (b - a)
    f1, f2 = phi(x1), phi(x2)
    
    for it in range(max_iter):
        if (b - a) <= tol:
            break
            
        if f1 > f2:
            a, x1, f1 = x1, x2, f2
            x2 = a + TAU * (b - a)
            f2 = phi(x2)
        else:
            b, x2, f2 = x2, x1, f1
            x1 = a + (1 - TAU) * (b - a)
            f1 = phi(x1)
    
    alpha = (a + b) / 2
    return alpha, phi(alpha), it + 1, time.time() - inicio, (b - a) <= tol
