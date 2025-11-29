import numpy as np
import pandas as pd
from funcoes_teste import FUNCOES
from metodo_gradiente import metodo_gradiente
from metodo_newton import metodo_newton
from metodo_bfgs import metodo_bfgs
from metodo_polak_ribiere import metodo_polak_ribiere

def executar_teste(nome_func, idx_inicial, metodo_nome, busca='armijo'):
    """Executa um teste com um método de otimização"""
    config = FUNCOES[nome_func]
    f = config['f']
    grad = config['grad']
    hess = config.get('hess')
    x0 = config['x_inicial'][idx_inicial]
    x_otimo_teorico = config['x_otimo']
    caso = config['nome_casos'][idx_inicial]
    
    # Executar método
    if metodo_nome == 'gradiente':
        x_opt, f_opt, iters, tempo, conv, hist = metodo_gradiente(f, grad, x0, busca_linear=busca)
    elif metodo_nome == 'newton':
        x_opt, f_opt, iters, tempo, conv, hist = metodo_newton(f, grad, hess, x0, busca_linear=busca)
    elif metodo_nome == 'bfgs':
        x_opt, f_opt, iters, tempo, conv, hist = metodo_bfgs(f, grad, x0, busca_linear=busca)
    elif metodo_nome == 'polak_ribiere':
        x_opt, f_opt, iters, tempo, conv, hist = metodo_polak_ribiere(f, grad, x0, busca_linear=busca)
    
    # Calcular erro
    erro = np.linalg.norm(x_opt - x_otimo_teorico)
    
    return {
        'funcao': nome_func,
        'caso': caso,
        'metodo': metodo_nome,
        'busca_linear': busca,
        'x0': x0,
        'x_otimo': x_opt,
        'f_otimo': f_opt,
        'x_teorico': x_otimo_teorico,
        'erro_x': erro,
        'iteracoes': iters,
        'tempo': tempo,
        'convergiu': conv,
        'f_inicial': f(x0),
        'reducao': f(x0) - f_opt
    }

def main():
    """Executa todos os testes"""
    print("="*80)
    print("OTIMIZAÇÃO IRRESTRITA - GRUPO 04")
    print("="*80)
    
    metodos = ['gradiente', 'newton', 'bfgs', 'polak_ribiere']
    resultados = []
    
    total = len(FUNCOES) * 2 * len(metodos)  # 5 funções × 2 pontos × 4 métodos = 40
    contador = 0
    
    for nome_func in sorted(FUNCOES.keys()):
        print(f"\n{nome_func}:")
        
        for idx_caso in range(2):
            caso = FUNCOES[nome_func]['nome_casos'][idx_caso]
            x0 = FUNCOES[nome_func]['x_inicial'][idx_caso]
            print(f"  {caso}: x0={x0}")
            
            for metodo in metodos:
                contador += 1
                try:
                    r = executar_teste(nome_func, idx_caso, metodo, busca='armijo')
                    resultados.append(r)
                    
                    print(f"    [{contador:2d}/{total}] [{metodo:15s}] "
                          f"f*={r['f_otimo']:.4e} erro={r['erro_x']:.4e} "
                          f"iters={r['iteracoes']:3d} {'✓' if r['convergiu'] else '✗'}")
                except Exception as e:
                    print(f"    [{contador:2d}/{total}] [{metodo:15s}] ERRO: {e}")
    
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
        erro_med = np.mean([r['erro_x'] for r in res])
        print(f"{metodo:15s}: {conv:2d}/{len(res)} convergiu, "
              f"iters={iters_med:5.1f}, erro={erro_med:.4e}")

if __name__ == "__main__":
    main()
