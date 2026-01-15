"""Script de teste simplificado - Testa apenas Problema 1"""
import numpy as np
import warnings

# Suprimir warnings numéricos
warnings.filterwarnings('ignore', category=RuntimeWarning)

from funcoes_teste import PROBLEMAS
from metodo_penalidade import metodo_penalidade
from metodo_lagrangeana_aumentada import metodo_lagrangeana_aumentada

print("="*80)
print("TESTE SIMPLIFICADO - PROBLEMA 1")
print("="*80)

# Carregar Problema 1
config = PROBLEMAS['Problema_1']
f = config['f']
grad_f = config['grad']
h_eq = config['h_eq']
grad_h_eq = config['grad_h_eq']

# Testar apenas o primeiro caso
x0 = config['x_inicial'][0]
caso = config['nome_casos'][0]

print(f"\n{caso}: x0={x0}")

# Teste 1: Lagrangeana Aumentada
print("\n1. Método de Lagrangeana Aumentada:")
try:
    x_opt, f_opt, iters, tempo, conv, hist = metodo_lagrangeana_aumentada(
        f, grad_f, x0,
        h_eq=h_eq,
        grad_h_eq=grad_h_eq,
        mu=10.0,
        tol=1e-6,
        max_iter_ext=30
    )

    violacao = abs(h_eq[0](x_opt))

    print(f"   x* = {x_opt}")
    print(f"   f* = {f_opt:.6e}")
    print(f"   Violação: {violacao:.4e}")
    print(f"   Iterações: {iters}")
    print(f"   Tempo: {tempo:.3f}s")
    print(f"   Convergiu: {'✓' if conv else '✗'}")
except Exception as e:
    print(f"   ERRO: {e}")

# Teste 2: Penalidade (β=10)
print("\n2. Método de Penalidade (β=10):")
try:
    x_opt, f_opt, iters, tempo, conv, hist = metodo_penalidade(
        f, grad_f, x0,
        h_eq=h_eq,
        grad_h_eq=grad_h_eq,
        beta=10,
        tol=1e-6,
        max_iter_ext=30
    )

    violacao = abs(h_eq[0](x_opt))

    print(f"   x* = {x_opt}")
    print(f"   f* = {f_opt:.6e}")
    print(f"   Violação: {violacao:.4e}")
    print(f"   Iterações: {iters}")
    print(f"   Tempo: {tempo:.3f}s")
    print(f"   Convergiu: {'✓' if conv else '✗'}")
except Exception as e:
    print(f"   ERRO: {e}")

print("\n" + "="*80)
print("TESTE CONCLUÍDO!")
print("="*80)
print("\nPara executar todos os testes, use:")
print("  python main.py")
print("\n(Isso executará 48 testes e pode demorar alguns minutos)")
