"""Método de Interpolação Quadrática - Grupo 04"""
import time

def interpolacao(phi, phi_derivada, alpha0=1.0, tol=1e-5, max_iter=100):
    """
    Interpolação Quadrática (com derivada)
    
    Retorna: (alpha_otimo, f_otimo, iteracoes, tempo, convergiu)
    """
    inicio = time.time()
    
    phi0 = phi(0)
    dphi0 = phi_derivada(0)
    
    if dphi0 >= 0:  # Não é direção de descida
        return 0, phi0, 0, time.time() - inicio, False
    
    alpha = alpha0
    
    for it in range(max_iter):
        if abs(alpha) < 1e-12:
            alpha = alpha0
            continue
            
        phi_alpha = phi(alpha)
        a = (phi_alpha - phi0 - dphi0 * alpha) / (alpha**2)
        
        if abs(a) < 1e-12 or a < 0:
            alpha /= 2
            continue
        
        alpha_novo = -dphi0 / (2 * a)
        
        if abs(alpha_novo - alpha) < tol:
            alpha = alpha_novo
            break
        
        if alpha_novo <= 0:
            alpha_novo = alpha / 2
        elif alpha_novo > 10 * alpha:
            alpha_novo = min(alpha_novo, 10 * alpha)
        
        alpha = alpha_novo
    
    return alpha, phi(alpha), it + 1, time.time() - inicio, it < max_iter
