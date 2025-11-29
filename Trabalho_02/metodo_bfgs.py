import numpy as np
import time
import sys
sys.path.append('../Trabalho_01')
from metodo_wolfe import condicoes_wolfe
from metodo_secao_aurea import secao_aurea
from utils import criar_phi, criar_phi_derivada

def metodo_bfgs(f, grad_f, x0, tol=1e-6, max_iter=1000, busca_linear='armijo'):
    """
    Método BFGS (Quase-Newton)
    Aproxima H^{-1} usando atualizações de rank-2
    
    Retorna: (x_otimo, f_otimo, iters, tempo, convergiu, historico)
    """
    inicio = time.time()
    
    n = len(x0)
    x = x0.copy()
    B = np.eye(n)  # Aproximação inicial de H^{-1}
    
    historico = {'x': [x.copy()], 'f': [f(x)], 'grad_norm': []}
    
    for k in range(max_iter):
        grad = grad_f(x)
        grad_norm = np.linalg.norm(grad)
        historico['grad_norm'].append(grad_norm)
        
        if grad_norm < tol:
            return x, f(x), k, time.time() - inicio, True, historico
        
        # Direção BFGS
        d = -B @ grad
        
        # Garantir direção de descida
        if np.dot(grad, d) >= 0:
            d = -grad
            B = np.eye(n)  # Reset
        
        # Busca linear
        phi = criar_phi(f, x, d)
        phi_derivada = criar_phi_derivada(f, grad_f, x, d)
        
        if busca_linear == 'armijo':
            alpha, _, _, _, _ = condicoes_wolfe(phi, phi_derivada, alpha0=1.0)
        else:  # secao_aurea
            alpha, _, _, _, _ = secao_aurea(phi, a=0, b=10)
        
        # Atualização do ponto
        s = alpha * d
        x_new = x + s
        grad_new = grad_f(x_new)
        
        # Atualização BFGS de B (aproximação de H^{-1})
        y = grad_new - grad
        
        sy = np.dot(s, y)
        if sy > 1e-10:  # Garantir curvatura positiva
            Bs = B @ s
            sBs = np.dot(s, Bs)
            
            B = B + np.outer(s, s) * (sy + sBs) / (sy**2) - (np.outer(Bs, s) + np.outer(s, Bs)) / sy
        
        x = x_new
        historico['x'].append(x.copy())
        historico['f'].append(f(x))
    
    return x, f(x), max_iter, time.time() - inicio, False, historico
