"""Condições de Wolfe - Grupo 04"""
import time

def condicoes_wolfe(phi, phi_derivada, alpha0=1.0, c1=1e-4, c2=0.9, 
                    alpha_max=10.0, max_iter=100):
    """
    Condições de Wolfe (inexato com derivada)
    
    Condições:
    1. Armijo: φ(α) ≤ φ(0) + c1*α*φ'(0)
    2. Curvatura: |φ'(α)| ≤ c2*|φ'(0)|
    
    Retorna: (alpha_otimo, f_otimo, iteracoes, tempo, convergiu)
    """
    inicio = time.time()
    
    phi0 = phi(0)
    dphi0 = phi_derivada(0)
    
    if dphi0 >= 0:  # Não é direção de descida
        return 0, phi0, 0, time.time() - inicio, False
    
    alpha = alpha0
    alpha_min = 0
    alpha_max_local = alpha_max
    
    for it in range(max_iter):
        phi_alpha = phi(alpha)
        dphi_alpha = phi_derivada(alpha)
        
        armijo = phi_alpha <= phi0 + c1 * alpha * dphi0
        curvatura = abs(dphi_alpha) <= c2 * abs(dphi0)
        
        if armijo and curvatura:
            return alpha, phi_alpha, it + 1, time.time() - inicio, True
        
        if not armijo:
            alpha_max_local = alpha
            alpha = (alpha_min + alpha_max_local) / 2
        elif dphi_alpha < 0:
            alpha_min = alpha
            alpha = (alpha_min + alpha_max_local) / 2 if alpha_max_local < float('inf') else 2 * alpha
        else:
            alpha_max_local = alpha
            alpha = (alpha_min + alpha_max_local) / 2
        
        alpha = max(1e-10, min(alpha, alpha_max))
    
    return alpha, phi(alpha), max_iter, time.time() - inicio, False

wolfe = condicoes_wolfe
