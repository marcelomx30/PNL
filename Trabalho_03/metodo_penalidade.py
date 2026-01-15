import numpy as np
import time
import sys
sys.path.append('../Trabalho_02')
from metodo_bfgs import metodo_bfgs

def metodo_penalidade(f, grad_f, x0, h_eq=None, grad_h_eq=None,
                      g_ineq=None, grad_g_ineq=None, bounds=None,
                      beta=10, tol=1e-6, max_iter_ext=50, max_iter_int=1000):
    """
    Método de Penalidade para otimização restrita

    Minimiza: f(x)
    Sujeito a: h_i(x) = 0 (restrições de igualdade)
               g_j(x) >= 0 (restrições de desigualdade)
               bounds[0] <= x_i <= bounds[1] (restrições de caixa)

    Parâmetros:
    -----------
    f : função objetivo
    grad_f : gradiente de f
    x0 : ponto inicial
    h_eq : lista de funções de restrição de igualdade
    grad_h_eq : lista de gradientes das restrições de igualdade
    g_ineq : lista de funções de restrição de desigualdade (formato g >= 0)
    grad_g_ineq : lista de gradientes das restrições de desigualdade
    bounds : tupla (lower, upper) para restrições de caixa
    beta : fator de crescimento do parâmetro de penalidade (default: 10)
    tol : tolerância para convergência
    max_iter_ext : número máximo de iterações externas
    max_iter_int : número máximo de iterações internas (BFGS)

    Retorna:
    --------
    x_opt : ponto ótimo
    f_opt : valor da função objetivo no ponto ótimo
    iters : número de iterações externas
    tempo : tempo de execução
    convergiu : booleano indicando convergência
    historico : dicionário com histórico da otimização
    """
    inicio = time.time()

    # Inicialização
    x = x0.copy()
    mu = 1.0  # Parâmetro de penalidade inicial

    # Ajustar x0 para satisfazer bounds se necessário
    if bounds is not None:
        x = np.clip(x, bounds[0], bounds[1])

    historico = {
        'x': [x.copy()],
        'f': [f(x)],
        'mu': [mu],
        'penalidade': [],
        'grad_norm': []
    }

    # Listas vazias se não fornecidas
    if h_eq is None:
        h_eq = []
        grad_h_eq = []
    if g_ineq is None:
        g_ineq = []
        grad_g_ineq = []

    def funcao_penalidade(x):
        """Função objetivo penalizada"""
        val = f(x)

        # Penalidade para restrições de igualdade: h_i(x)^2
        for h in h_eq:
            val += (mu / 2) * h(x)**2

        # Penalidade para restrições de desigualdade: [min(0, g_j(x))]^2
        for g in g_ineq:
            val += (mu / 2) * min(0, g(x))**2

        # Penalidade para bounds
        if bounds is not None:
            for i in range(len(x)):
                # Penalidade para limite inferior: xi >= bounds[0]
                if x[i] < bounds[0]:
                    val += (mu / 2) * (bounds[0] - x[i])**2
                # Penalidade para limite superior: xi <= bounds[1]
                if x[i] > bounds[1]:
                    val += (mu / 2) * (x[i] - bounds[1])**2

        return val

    def grad_penalidade(x):
        """Gradiente da função penalizada"""
        grad = grad_f(x).copy()

        # Gradiente das penalidades de igualdade
        for h, grad_h in zip(h_eq, grad_h_eq):
            grad += mu * h(x) * grad_h(x)

        # Gradiente das penalidades de desigualdade
        for g, grad_g in zip(g_ineq, grad_g_ineq):
            if g(x) < 0:
                grad -= mu * g(x) * grad_g(x)

        # Gradiente das penalidades de bounds
        if bounds is not None:
            for i in range(len(x)):
                if x[i] < bounds[0]:
                    grad[i] -= mu * (bounds[0] - x[i])
                if x[i] > bounds[1]:
                    grad[i] += mu * (x[i] - bounds[1])

        return grad

    # Iterações externas (aumento de mu)
    for k in range(max_iter_ext):
        # Resolver subproblema irrestrito com BFGS
        try:
            x_new, f_pen, _, _, conv_bfgs, _ = metodo_bfgs(
                funcao_penalidade,
                grad_penalidade,
                x,
                tol=tol/10,  # Tolerância mais rigorosa no subproblema
                max_iter=max_iter_int
            )
        except Exception as e:
            # Erro silencioso - continua com x atual
            break

        # Calcular violação das restrições
        violacao_eq = sum(abs(h(x_new)) for h in h_eq)
        violacao_ineq = sum(abs(min(0, g(x_new))) for g in g_ineq)

        # Violação de bounds
        violacao_bounds = 0
        if bounds is not None:
            for i in range(len(x_new)):
                if x_new[i] < bounds[0]:
                    violacao_bounds += abs(bounds[0] - x_new[i])
                if x_new[i] > bounds[1]:
                    violacao_bounds += abs(x_new[i] - bounds[1])

        violacao_total = violacao_eq + violacao_ineq + violacao_bounds

        # Calcular penalidade total
        penalidade = 0
        for h in h_eq:
            penalidade += h(x_new)**2
        for g in g_ineq:
            penalidade += min(0, g(x_new))**2
        if bounds is not None:
            for i in range(len(x_new)):
                if x_new[i] < bounds[0]:
                    penalidade += (bounds[0] - x_new[i])**2
                if x_new[i] > bounds[1]:
                    penalidade += (x_new[i] - bounds[1])**2

        # Atualizar histórico
        historico['x'].append(x_new.copy())
        historico['f'].append(f(x_new))
        historico['mu'].append(mu)
        historico['penalidade'].append(penalidade)
        historico['grad_norm'].append(np.linalg.norm(grad_f(x_new)))

        # Verificar convergência
        if violacao_total < tol and np.linalg.norm(x_new - x) < tol:
            tempo_total = time.time() - inicio
            return x_new, f(x_new), k+1, tempo_total, True, historico

        # Atualizar x e aumentar mu
        x = x_new
        mu = mu * beta

        # Verificar se mu está ficando muito grande (problema mal condicionado)
        if mu > 1e10:
            # Parar silenciosamente quando mu muito grande
            break

    # Não convergiu
    tempo_total = time.time() - inicio
    return x, f(x), max_iter_ext, tempo_total, False, historico
