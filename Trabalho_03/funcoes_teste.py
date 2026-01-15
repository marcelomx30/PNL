import numpy as np

# ==================== PROBLEMA 1 ====================
def f1(x):
    """f1(x) = 0.01(x1-1)^2 + (x2-x1^2)^2
    Note: x tem 3 componentes (x1, x2, x3), mas x3 aparece apenas na restrição"""
    return 0.01*(x[0] - 1)**2 + (x[1] - x[0]**2)**2

def grad_f1(x):
    """Gradiente de f1 (3D, mas derivada em x3 é zero)"""
    df1 = 0.02*(x[0] - 1) - 4*x[0]*(x[1] - x[0]**2)
    df2 = 2*(x[1] - x[0]**2)
    df3 = 0.0  # Derivada parcial em relação a x3 é zero
    return np.array([df1, df2, df3])

def hess_f1(x):
    """Hessiana de f1 (3x3)"""
    h11 = 0.02 - 4*x[1] + 12*x[0]**2
    h12 = -4*x[0]
    h21 = -4*x[0]
    h22 = 2
    # Terceira linha e coluna são zeros (sem dependência de x3)
    return np.array([[h11, h12, 0.],
                     [h21, h22, 0.],
                     [0.,  0.,  0.]])

def h1(x):
    """Restrição de igualdade: h(x) = x1 + x3^2 - 1 = 0"""
    return x[0] + x[2]**2 - 1

def grad_h1(x):
    """Gradiente da restrição h1"""
    return np.array([1., 0., 2*x[2]])

def hess_h1(x):
    """Hessiana da restrição h1"""
    return np.array([[0., 0., 0.],
                     [0., 0., 0.],
                     [0., 0., 2.]])

# ==================== PROBLEMA 2 ====================
def f2(x):
    """f2(x) = x1*x4*(x1 + x2 + x3) + x3"""
    return x[0]*x[3]*(x[0] + x[1] + x[2]) + x[2]

def grad_f2(x):
    """Gradiente de f2"""
    df1 = x[3]*(2*x[0] + x[1] + x[2])
    df2 = x[0]*x[3]
    df3 = x[0]*x[3] + 1
    df4 = x[0]*(x[0] + x[1] + x[2])
    return np.array([df1, df2, df3, df4])

def hess_f2(x):
    """Hessiana de f2"""
    H = np.zeros((4, 4))
    H[0, 0] = 2*x[3]
    H[0, 1] = H[1, 0] = x[3]
    H[0, 2] = H[2, 0] = x[3]
    H[0, 3] = H[3, 0] = 2*x[0] + x[1] + x[2]
    H[1, 3] = H[3, 1] = x[0]
    H[2, 3] = H[3, 2] = x[0]
    return H

def g2_1(x):
    """Restrição de desigualdade: g1(x) = 25 - x1*x2*x3*x4 >= 0"""
    return 25 - x[0]*x[1]*x[2]*x[3]

def grad_g2_1(x):
    """Gradiente de g2_1"""
    return np.array([-x[1]*x[2]*x[3], -x[0]*x[2]*x[3],
                     -x[0]*x[1]*x[3], -x[0]*x[1]*x[2]])

def g2_2(x):
    """Restrição de desigualdade: g2(x) = 40 - (x1^2 + x2^2 + x3^2 + x4^2) >= 0"""
    return 40 - np.sum(x**2)

def grad_g2_2(x):
    """Gradiente de g2_2"""
    return -2*x

def g2_bounds_lower(x):
    """Restrições de caixa inferior: xi >= 1, transformadas em gi(x) = xi - 1 >= 0"""
    return x - 1

def g2_bounds_upper(x):
    """Restrições de caixa superior: xi <= 5, transformadas em gi(x) = 5 - xi >= 0"""
    return 5 - x

# ==================== PROBLEMA 3 ====================
def f3(x):
    """f3(x) = (x1^2 + x2 - 11)^2 + (x1 + x2^2 - 7)^2"""
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def grad_f3(x):
    """Gradiente de f3"""
    df1 = 4*x[0]*(x[0]**2 + x[1] - 11) + 2*(x[0] + x[1]**2 - 7)
    df2 = 2*(x[0]**2 + x[1] - 11) + 4*x[1]*(x[0] + x[1]**2 - 7)
    return np.array([df1, df2])

def hess_f3(x):
    """Hessiana de f3"""
    h11 = 12*x[0]**2 + 4*x[1] - 42
    h12 = 4*x[0] + 4*x[1]
    h21 = h12
    h22 = 4*x[0] + 12*x[1]**2 - 26
    return np.array([[h11, h12], [h21, h22]])

def g3(x):
    """Restrição de desigualdade: g(x) = (x1-3)^2 + (x2-2)^2 - 25 <= 0"""
    return (x[0] - 3)**2 + (x[1] - 2)**2 - 25

def grad_g3(x):
    """Gradiente de g3"""
    return np.array([2*(x[0] - 3), 2*(x[1] - 2)])

def hess_g3(x):
    """Hessiana de g3"""
    return np.array([[2., 0.], [0., 2.]])

# ==================== PROBLEMA 4 ====================
def f4(x):
    """f4(x) = (x1-2)^2 + (x2-1)^2"""
    return (x[0] - 2)**2 + (x[1] - 1)**2

def grad_f4(x):
    """Gradiente de f4"""
    return np.array([2*(x[0] - 2), 2*(x[1] - 1)])

def hess_f4(x):
    """Hessiana de f4"""
    return np.array([[2., 0.], [0., 2.]])

def g4(x):
    """Restrição de desigualdade: g(x) = 0.25*x1^2 + x2^2 - 1 <= 0"""
    return 0.25*x[0]**2 + x[1]**2 - 1

def grad_g4(x):
    """Gradiente de g4"""
    return np.array([0.5*x[0], 2*x[1]])

def hess_g4(x):
    """Hessiana de g4"""
    return np.array([[0.5, 0.], [0., 2.]])

def h4(x):
    """Restrição de igualdade: h(x) = x1 - 2*x2 + 1 = 0"""
    return x[0] - 2*x[1] + 1

def grad_h4(x):
    """Gradiente de h4"""
    return np.array([1., -2.])

def hess_h4(x):
    """Hessiana de h4"""
    return np.array([[0., 0.], [0., 0.]])

# ==================== CONFIGURAÇÃO DOS TESTES ====================
PROBLEMAS = {
    'Problema_1': {
        'f': f1,
        'grad': grad_f1,
        'hess': hess_f1,
        'h_eq': [h1],  # Lista de restrições de igualdade
        'grad_h_eq': [grad_h1],
        'hess_h_eq': [hess_h1],
        'g_ineq': [],  # Lista de restrições de desigualdade
        'grad_g_ineq': [],
        'hess_g_ineq': [],
        'x_inicial': [
            np.array([-2., 2., 0.]),
            np.array([5., 0., 1.]),
            np.array([-1.01, 0., 0.01])
        ],
        'nome_casos': ['caso_i', 'caso_ii', 'caso_iii'],
        'dimensao': 3
    },
    'Problema_2': {
        'f': f2,
        'grad': grad_f2,
        'hess': hess_f2,
        'h_eq': [],
        'grad_h_eq': [],
        'hess_h_eq': [],
        'g_ineq': [g2_1, g2_2],  # g1 e g2
        'grad_g_ineq': [grad_g2_1, grad_g2_2],
        'hess_g_ineq': [],
        'x_inicial': [
            np.array([4., 4., 4., 4.]),
            np.array([5., 5., 5., 5.]),
            np.array([2., 3., 4., 5.])
        ],
        'bounds': (1, 5),  # 1 <= xi <= 5
        'nome_casos': ['caso_i', 'caso_ii', 'caso_iii'],
        'dimensao': 4
    },
    'Problema_3': {
        'f': f3,
        'grad': grad_f3,
        'hess': hess_f3,
        'h_eq': [],
        'grad_h_eq': [],
        'hess_h_eq': [],
        'g_ineq': [lambda x: -g3(x)],  # g(x) <= 0 -> -g(x) >= 0
        'grad_g_ineq': [lambda x: -grad_g3(x)],
        'hess_g_ineq': [lambda x: -hess_g3(x)],
        'x_inicial': [
            np.array([3., 2.]),
            np.array([0., 0.]),
            np.array([6., 5.])
        ],
        'nome_casos': ['caso_i', 'caso_ii', 'caso_iii'],
        'dimensao': 2
    },
    'Problema_4': {
        'f': f4,
        'grad': grad_f4,
        'hess': hess_f4,
        'h_eq': [h4],
        'grad_h_eq': [grad_h4],
        'hess_h_eq': [hess_h4],
        'g_ineq': [lambda x: -g4(x)],  # g(x) <= 0 -> -g(x) >= 0
        'grad_g_ineq': [lambda x: -grad_g4(x)],
        'hess_g_ineq': [lambda x: -hess_g4(x)],
        'x_inicial': [
            np.array([-1., 0.5]),
            np.array([-0.6, 0.25]),
            np.array([-0.4, 0.4])
        ],
        'nome_casos': ['caso_i', 'caso_ii', 'caso_iii'],
        'dimensao': 2
    }
}
