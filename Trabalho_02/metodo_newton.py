import numpy as np
import time
import sys
sys.path.append('../Trabalho_01')
from metodo_wolfe import condicoes_wolfe
from metodo_secao_aurea import secao_aurea
from utils import criar_phi, criar_phi_derivada

def metodo_newton(f, grad_f, hess_f, x0, tol=1e-6, max_iter=1000, busca_linear='armijo'):
    """
    Método de Newton
    d_k = -H^{-1}(x_k) * ∇f(x_k)
    
    Retorna: (x_otimo, f_otimo, iters, tempo, convergiu, historico)
    """
    inicio = time.time()
    
    x = x0.copy()
    historico = {'x': [x.copy()], 'f': [f(x)], 'grad_norm': []}
    
    for k in range(max_iter):
        grad = grad_f(x)
        grad_norm = np.linalg.norm(grad)
        historico['grad_norm'].append(grad_norm)
        
        if grad_norm < tol:
            return x, f(x), k, time.time() - inicio, True, historico
        
        # Direção de Newton
        H = hess_f(x)
        try:
            d = -np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            # Hessiana singular, usar gradiente
            d = -grad
        
        # Garantir direção de descida
        if np.dot(grad, d) >= 0:
            d = -grad
        
        # Busca linear
        phi = criar_phi(f, x, d)
        phi_derivada = criar_phi_derivada(f, grad_f, x, d)
        
        if busca_linear == 'armijo':
            alpha, _, _, _, _ = condicoes_wolfe(phi, phi_derivada, alpha0=1.0)
        else:  # secao_aurea
            alpha, _, _, _, _ = secao_aurea(phi, a=0, b=10)
        
        # Atualização
        x = x + alpha * d
        historico['x'].append(x.copy())
        historico['f'].append(f(x))
    
    return x, f(x), max_iter, time.time() - inicio, False, historico
