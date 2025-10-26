"""Programa Principal - Métodos de Busca Linear - Grupo 04"""
import numpy as np
import pandas as pd
from funcoes_teste import FUNCOES, CASOS_TESTE
from utils import criar_phi, criar_phi_derivada
from metodo_secao_aurea import secao_aurea
from metodo_interpolacao import interpolacao
from metodo_wolfe import condicoes_wolfe

def executar_teste(nome_func, caso, metodo_nome):
    """Executa um teste"""
    f, grad_f = FUNCOES[nome_func]
    x_k = caso['x']
    d_k = -grad_f(x_k)
    
    phi = criar_phi(f, x_k, d_k)
    phi_derivada = criar_phi_derivada(f, grad_f, x_k, d_k)
    
    if metodo_nome == 'secao_aurea':
        alpha, f_alpha, iters, tempo, conv = secao_aurea(phi, a=0, b=10)
    elif metodo_nome == 'interpolacao':
        alpha, f_alpha, iters, tempo, conv = interpolacao(phi, phi_derivada)
    elif metodo_nome == 'wolfe':
        alpha, f_alpha, iters, tempo, conv = condicoes_wolfe(phi, phi_derivada)
    
    x_novo = x_k + alpha * d_k
    
    return {
        'funcao': nome_func,
        'caso': caso['nome'],
        'metodo': metodo_nome,
        'f_inicial': f(x_k),
        'alpha_otimo': alpha,
        'phi_alpha': f_alpha,
        'f_x_novo': f(x_novo),
        'iteracoes': iters,
        'tempo': tempo,
        'convergiu': conv
    }

def main():
    """Executa todos os testes"""
    print("="*80)
    print("MÉTODOS DE BUSCA LINEAR - GRUPO 04")
    print("="*80)
    
    metodos = ['secao_aurea', 'interpolacao', 'wolfe']
    resultados = []
    
    for nome_func in sorted(CASOS_TESTE.keys()):
        print(f"\n{nome_func}:")
        for caso in CASOS_TESTE[nome_func]:
            print(f"  {caso['nome']}: x={caso['x']}")
            for metodo in metodos:
                try:
                    r = executar_teste(nome_func, caso, metodo)
                    resultados.append(r)
                    print(f"    [{metodo:15s}] α*={r['alpha_otimo']:.6f} "
                          f"f={r['f_x_novo']:.4e} iters={r['iteracoes']:3d} "
                          f"{'✓' if r['convergiu'] else '✗'}")
                except Exception as e:
                    print(f"    [{metodo:15s}] ERRO: {e}")
    
    # Salvar resultados
    df = pd.DataFrame(resultados)
    df.to_csv('resultados.csv', index=False)
    print(f"\n✓ Resultados salvos em 'resultados.csv'")
    
    # Resumo
    print("\n" + "="*80)
    print("RESUMO")
    print("="*80)
    for metodo in metodos:
        res = [r for r in resultados if r['metodo'] == metodo]
        conv = sum(1 for r in res if r['convergiu'])
        iters_med = np.mean([r['iteracoes'] for r in res])
        print(f"{metodo:15s}: {conv}/{len(res)} convergiu, "
              f"{iters_med:.1f} iters média")

if __name__ == "__main__":
    main()
