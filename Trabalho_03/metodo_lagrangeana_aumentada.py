import numpy as np
import time
import sys
sys.path.append('../Trabalho_02')
from metodo_bfgs import metodo_bfgs

def metodo_lagrangeana_aumentada(f, grad_f, x0, h_eq=None, grad_h_eq=None,
                                 g_ineq=None, grad_g_ineq=None, bounds=None,
                                 mu=10.0, tol=1e-6, max_iter_ext=50, max_iter_int=1000):
    """
    Método da Lagrangeana Aumentada para otimização restrita

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
    mu : parâmetro de penalidade (default: 10.0)
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

    # Ajustar x0 para satisfazer bounds se necessário
    if bounds is not None:
        x = np.clip(x, bounds[0], bounds[1])

    # Listas vazias se não fornecidas
    if h_eq is None:
        h_eq = []
        grad_h_eq = []
    if g_ineq is None:
        g_ineq = []
        grad_g_ineq = []

    # Inicializar multiplicadores de Lagrange
    lambda_eq = [0.0] * len(h_eq)  # Para restrições de igualdade
    lambda_ineq = [0.0] * len(g_ineq)  # Para restrições de desigualdade

    # Multiplicadores para bounds
    lambda_bounds_lower = np.zeros(len(x))
    lambda_bounds_upper = np.zeros(len(x))

    historico = {
        'x': [x.copy()],
        'f': [f(x)],
        'lambda_eq': [lambda_eq.copy()],
        'lambda_ineq': [lambda_ineq.copy()],
        'mu': [mu],
        'grad_norm': []
    }

    def lagrangeana_aumentada(x):
        """Função Lagrangeana Aumentada"""
        val = f(x)

        # Termo da Lagrangeana para restrições de igualdade
        # L = f(x) - sum(lambda_i * h_i(x)) + (mu/2) * sum(h_i(x)^2)
        for i, h in enumerate(h_eq):
            h_val = h(x)
            val -= lambda_eq[i] * h_val
            val += (mu / 2) * h_val**2

        # Termo da Lagrangeana para restrições de desigualdade
        # L = f(x) - sum(lambda_j * g_j(x)) + (mu/2) * [max(0, -g_j(x) + lambda_j/mu)]^2
        for i, g in enumerate(g_ineq):
            g_val = g(x)
            val -= lambda_ineq[i] * g_val
            val += (mu / 2) * max(0, -g_val + lambda_ineq[i]/mu)**2

        # Termos para bounds
        if bounds is not None:
            for i in range(len(x)):
                # Limite inferior: xi >= bounds[0]
                g_lower = x[i] - bounds[0]
                val -= lambda_bounds_lower[i] * g_lower
                val += (mu / 2) * max(0, -g_lower + lambda_bounds_lower[i]/mu)**2

                # Limite superior: xi <= bounds[1] -> bounds[1] - xi >= 0
                g_upper = bounds[1] - x[i]
                val -= lambda_bounds_upper[i] * g_upper
                val += (mu / 2) * max(0, -g_upper + lambda_bounds_upper[i]/mu)**2

        return val

    def grad_lagrangeana(x):
        """Gradiente da Lagrangeana Aumentada"""
        grad = grad_f(x).copy()

        # Gradiente para restrições de igualdade
        for i, (h, grad_h) in enumerate(zip(h_eq, grad_h_eq)):
            h_val = h(x)
            grad -= lambda_eq[i] * grad_h(x)
            grad += mu * h_val * grad_h(x)

        # Gradiente para restrições de desigualdade
        for i, (g, grad_g) in enumerate(zip(g_ineq, grad_g_ineq)):
            g_val = g(x)
            grad -= lambda_ineq[i] * grad_g(x)
            viol = max(0, -g_val + lambda_ineq[i]/mu)
            if viol > 0:
                grad -= mu * viol * grad_g(x)

        # Gradiente para bounds
        if bounds is not None:
            for i in range(len(x)):
                # Limite inferior
                g_lower = x[i] - bounds[0]
                viol_lower = max(0, -g_lower + lambda_bounds_lower[i]/mu)
                if viol_lower > 0:
                    grad[i] -= mu * viol_lower

                # Limite superior
                g_upper = bounds[1] - x[i]
                viol_upper = max(0, -g_upper + lambda_bounds_upper[i]/mu)
                if viol_upper > 0:
                    grad[i] += mu * viol_upper

        return grad

    # Iterações externas
    for k in range(max_iter_ext):
        # Resolver subproblema irrestrito com BFGS
        try:
            x_new, f_lag, _, _, conv_bfgs, _ = metodo_bfgs(
                lagrangeana_aumentada,
                grad_lagrangeana,
                x,
                tol=tol/10,
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

        # Atualizar multiplicadores de Lagrange
        # Para restrições de igualdade: lambda_i := lambda_i - mu * h_i(x)
        for i, h in enumerate(h_eq):
            lambda_eq[i] -= mu * h(x_new)

        # Para restrições de desigualdade: lambda_i := max(0, lambda_i - mu * g_i(x))
        for i, g in enumerate(g_ineq):
            lambda_ineq[i] = max(0, lambda_ineq[i] - mu * g(x_new))

        # Para bounds
        if bounds is not None:
            for i in range(len(x_new)):
                # Limite inferior
                g_lower = x_new[i] - bounds[0]
                lambda_bounds_lower[i] = max(0, lambda_bounds_lower[i] - mu * g_lower)

                # Limite superior
                g_upper = bounds[1] - x_new[i]
                lambda_bounds_upper[i] = max(0, lambda_bounds_upper[i] - mu * g_upper)

        # Atualizar histórico
        historico['x'].append(x_new.copy())
        historico['f'].append(f(x_new))
        historico['lambda_eq'].append(lambda_eq.copy())
        historico['lambda_ineq'].append(lambda_ineq.copy())
        historico['mu'].append(mu)
        historico['grad_norm'].append(np.linalg.norm(grad_f(x_new)))

        # Verificar convergência
        if violacao_total < tol and np.linalg.norm(x_new - x) < tol:
            tempo_total = time.time() - inicio
            return x_new, f(x_new), k+1, tempo_total, True, historico

        # Atualizar x
        x = x_new

        # Opcionalmente aumentar mu se a violação não está diminuindo
        # (estratégia conservadora: manter mu fixo)

    # Não convergiu
    tempo_total = time.time() - inicio
    return x, f(x), max_iter_ext, tempo_total, False, historico
