"""Funções Auxiliares - Projeto Busca Linear - Grupo 04"""
import numpy as np

def criar_phi(f, x_k, d_k):
    """Cria φ(α) = f(x_k + α*d_k)"""
    return lambda alpha: f(x_k + alpha * d_k)

def criar_phi_derivada(f, grad_f, x_k, d_k):
    """Cria φ'(α) = ∇f(x_k + α*d_k)^T · d_k"""
    return lambda alpha: np.dot(grad_f(x_k + alpha * d_k), d_k)
