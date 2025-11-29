import numpy as np
import time
import sys
sys.path.append('../Trabalho_01')
from metodo_wolfe import condicoes_wolfe
from metodo_secao_aurea import secao_aurea
from utils import criar_phi, criar_phi_derivada

def metodo_polak_ribiere(f, grad_f, x0, tol=1e-6, max_iter=1000, busca_linear='armijo'):
    """
    Método de Polak-Ribière (Gradiente Conjugado)
    β_k^PR = (∇f_{k+1}^T (∇f_{k+1} - ∇f_k)) / ||∇f_k||^2
    d_{k+1} = -∇f_{k+1} + β_k * d_k
    
    Retorna: (x_otimo, f_otimo, iters, tempo, convergiu, historico)
    """
    inicio = time.time()
    
    x = x0.copy()
    grad = grad_f(x)
    d = -grad  # Primeira direção é o gradiente negativo
    
    historico = {'x': [x.copy()], 'f': [f(x)], 'grad_norm': []}
    
    for k in range(max_iter):
        grad_norm = np.linalg.norm(grad)
        historico['grad_norm'].append(grad_norm)
        
        if grad_norm < tol:
            return x, f(x), k, time.time() - inicio, True, historico
        
        # Busca linear
        phi = criar_phi(f, x, d)
        phi_derivada = criar_phi_derivada(f, grad_f, x, d)
        
        if busca_linear == 'armijo':
            alpha, _, _, _, _ = condicoes_wolfe(phi, phi_derivada, alpha0=1.0)
        else:  # secao_aurea
            alpha, _, _, _, _ = secao_aurea(phi, a=0, b=10)
        
        # Atualização do ponto
        x_new = x + alpha * d
        grad_new = grad_f(x_new)
        
        # Cálculo de β (Polak-Ribière)
        beta_num = np.dot(grad_new, grad_new - grad)
        beta_den = np.dot(grad, grad)
        
        if beta_den < 1e-10:
            beta = 0
        else:
            beta = max(0, beta_num / beta_den)  # Polak-Ribière com reset
        
        # Nova direção conjugada
        d = -grad_new + beta * d
        
        # Garantir direção de descida
        if np.dot(grad_new, d) >= 0:
            d = -grad_new  # Reset para gradiente
        
        x = x_new
        grad = grad_new
        
        historico['x'].append(x.copy())
        historico['f'].append(f(x))
    
    return x, f(x), max_iter, time.time() - inicio, False, historico
