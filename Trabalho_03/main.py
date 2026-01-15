import numpy as np
import pandas as pd
import warnings

# Suprimir warnings numéricos (overflow, invalid value, etc.)
warnings.filterwarnings('ignore', category=RuntimeWarning)

from funcoes_teste import PROBLEMAS
from metodo_penalidade import metodo_penalidade
from metodo_lagrangeana_aumentada import metodo_lagrangeana_aumentada

def executar_teste(nome_prob, idx_inicial, metodo_nome, beta=10):
    """Executa um teste com um método de otimização restrita"""
    config = PROBLEMAS[nome_prob]
    f = config['f']
    grad = config['grad']
    x0 = config['x_inicial'][idx_inicial]
    caso = config['nome_casos'][idx_inicial]
    h_eq = config.get('h_eq', [])
    grad_h_eq = config.get('grad_h_eq', [])
    g_ineq = config.get('g_ineq', [])
    grad_g_ineq = config.get('grad_g_ineq', [])
    bounds = config.get('bounds', None)

    # Executar método
    if metodo_nome == 'penalidade':
        x_opt, f_opt, iters, tempo, conv, hist = metodo_penalidade(
            f, grad, x0,
            h_eq=h_eq, grad_h_eq=grad_h_eq,
            g_ineq=g_ineq, grad_g_ineq=grad_g_ineq,
            bounds=bounds,
            beta=beta
        )
    elif metodo_nome == 'lagrangeana':
        x_opt, f_opt, iters, tempo, conv, hist = metodo_lagrangeana_aumentada(
            f, grad, x0,
            h_eq=h_eq, grad_h_eq=grad_h_eq,
            g_ineq=g_ineq, grad_g_ineq=grad_g_ineq,
            bounds=bounds
        )

    # Calcular violação das restrições no ponto ótimo
    violacao_eq = sum(abs(h(x_opt)) for h in h_eq) if h_eq else 0.0
    violacao_ineq = sum(abs(min(0, g(x_opt))) for g in g_ineq) if g_ineq else 0.0
    violacao_bounds = 0.0
    if bounds is not None:
        for i in range(len(x_opt)):
            if x_opt[i] < bounds[0]:
                violacao_bounds += abs(bounds[0] - x_opt[i])
            if x_opt[i] > bounds[1]:
                violacao_bounds += abs(x_opt[i] - bounds[1])

    violacao_total = violacao_eq + violacao_ineq + violacao_bounds

    return {
        'problema': nome_prob,
        'caso': caso,
        'metodo': metodo_nome,
        'beta': beta if metodo_nome == 'penalidade' else None,
        'x0': x0,
        'x_otimo': x_opt,
        'f_otimo': f_opt,
        'f_inicial': f(x0),
        'reducao': f(x0) - f_opt,
        'violacao_eq': violacao_eq,
        'violacao_ineq': violacao_ineq,
        'violacao_bounds': violacao_bounds,
        'violacao_total': violacao_total,
        'iteracoes': iters,
        'tempo': tempo,
        'convergiu': conv
    }

def main():
    """Executa todos os testes"""
    print("="*80)
    print("OTIMIZAÇÃO RESTRITA - GRUPO 04")
    print("Métodos: Penalidade e Lagrangeana Aumentada")
    print("="*80)

    # Configuração dos métodos
    metodos = ['penalidade', 'lagrangeana']
    betas = [2, 10, 100]  # Diferentes taxas de crescimento para o método de penalidade

    resultados = []

    # Contador para progresso
    total_testes = 0
    for nome_prob in sorted(PROBLEMAS.keys()):
        num_casos = len(PROBLEMAS[nome_prob]['x_inicial'])
        # Para penalidade: 3 casos × 3 betas = 9 testes
        # Para lagrangeana: 3 casos × 1 = 3 testes
        total_testes += num_casos * (len(betas) + 1)

    contador = 0

    for nome_prob in sorted(PROBLEMAS.keys()):
        print(f"\n{nome_prob}:")
        config = PROBLEMAS[nome_prob]

        for idx_caso in range(len(config['x_inicial'])):
            caso = config['nome_casos'][idx_caso]
            x0 = config['x_inicial'][idx_caso]
            print(f"  {caso}: x0={x0}")

            # Testar Lagrangeana Aumentada (beta fixo)
            contador += 1
            try:
                r = executar_teste(nome_prob, idx_caso, 'lagrangeana')
                resultados.append(r)

                print(f"    [{contador:2d}/{total_testes}] [Lagrangeana     ] "
                      f"f*={r['f_otimo']:.6e} viol={r['violacao_total']:.4e} "
                      f"iters={r['iteracoes']:3d} {'✓' if r['convergiu'] else '✗'}")
            except Exception as e:
                print(f"    [{contador:2d}/{total_testes}] [Lagrangeana     ] ERRO: {e}")

            # Testar Penalidade com diferentes betas
            for beta in betas:
                contador += 1
                try:
                    r = executar_teste(nome_prob, idx_caso, 'penalidade', beta=beta)
                    resultados.append(r)

                    print(f"    [{contador:2d}/{total_testes}] [Penalidade β={beta:3d}] "
                          f"f*={r['f_otimo']:.6e} viol={r['violacao_total']:.4e} "
                          f"iters={r['iteracoes']:3d} {'✓' if r['convergiu'] else '✗'}")
                except Exception as e:
                    print(f"    [{contador:2d}/{total_testes}] [Penalidade β={beta:3d}] ERRO: {e}")

    # Salvar resultados
    df = pd.DataFrame(resultados)

    # Converter arrays numpy para strings para salvar no CSV
    df['x0'] = df['x0'].apply(lambda x: np.array2string(x, separator=', '))
    df['x_otimo'] = df['x_otimo'].apply(lambda x: np.array2string(x, separator=', ', precision=6))

    df.to_csv('resultados.csv', index=False)
    print(f"\n✓ Resultados salvos em 'resultados.csv'")

    # Resumo por método
    print("\n" + "="*80)
    print("RESUMO - LAGRANGEANA AUMENTADA")
    print("="*80)
    res_lag = [r for r in resultados if r['metodo'] == 'lagrangeana']
    if res_lag:
        conv = sum(1 for r in res_lag if r['convergiu'])
        iters_med = np.mean([r['iteracoes'] for r in res_lag])
        viol_med = np.mean([r['violacao_total'] for r in res_lag])
        print(f"Convergência: {conv}/{len(res_lag)}")
        print(f"Iterações médias: {iters_med:.1f}")
        print(f"Violação média: {viol_med:.4e}")

    print("\n" + "="*80)
    print("RESUMO - PENALIDADE (por β)")
    print("="*80)
    for beta in betas:
        res_pen = [r for r in resultados if r['metodo'] == 'penalidade' and r['beta'] == beta]
        if res_pen:
            conv = sum(1 for r in res_pen if r['convergiu'])
            iters_med = np.mean([r['iteracoes'] for r in res_pen])
            viol_med = np.mean([r['violacao_total'] for r in res_pen])
            print(f"β={beta:3d}: {conv:2d}/{len(res_pen)} convergiu, "
                  f"iters={iters_med:5.1f}, violação={viol_med:.4e}")

    # Análise comparativa
    print("\n" + "="*80)
    print("ANÁLISE COMPARATIVA")
    print("="*80)
    for nome_prob in sorted(PROBLEMAS.keys()):
        print(f"\n{nome_prob}:")
        res_prob = [r for r in resultados if r['problema'] == nome_prob]

        # Melhor resultado de cada método
        res_lag_prob = [r for r in res_prob if r['metodo'] == 'lagrangeana']
        res_pen_prob = [r for r in res_prob if r['metodo'] == 'penalidade']

        if res_lag_prob:
            melhor_lag = min(res_lag_prob, key=lambda x: x['violacao_total'])
            print(f"  Lagrangeana: f*={melhor_lag['f_otimo']:.6e}, "
                  f"viol={melhor_lag['violacao_total']:.4e}")

        if res_pen_prob:
            melhor_pen = min(res_pen_prob, key=lambda x: x['violacao_total'])
            print(f"  Penalidade (β={melhor_pen['beta']}): f*={melhor_pen['f_otimo']:.6e}, "
                  f"viol={melhor_pen['violacao_total']:.4e}")

if __name__ == "__main__":
    main()
