import numpy as np

# ==================== FUNÇÃO 1 ====================
def f1(x):
    """f1(x) = (x1-2)^4 + (x1-2*x2)^2"""
    return (x[0] - 2)**4 + (x[0] - 2*x[1])**2

def grad_f1(x):
    """Gradiente de f1"""
    df1 = 4*(x[0]-2)**3 + 2*(x[0]-2*x[1])
    df2 = -4*(x[0]-2*x[1])
    return np.array([df1, df2])

def hess_f1(x):
    """Hessiana de f1"""
    h11 = 12*(x[0]-2)**2 + 2
    h12 = -4
    h21 = -4
    h22 = 8
    return np.array([[h11, h12], [h21, h22]])

# ==================== FUNÇÃO 2 (Rosenbrock) ====================
def f2(x):
    """f2(x) = 100(x2-x1^2)^2 + (1-x1)^2"""
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

def grad_f2(x):
    """Gradiente de f2 (Rosenbrock)"""
    df1 = -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0])
    df2 = 200*(x[1] - x[0]**2)
    return np.array([df1, df2])

def hess_f2(x):
    """Hessiana de f2 (Rosenbrock)"""
    h11 = -400*(x[1] - 3*x[0]**2) + 2
    h12 = -400*x[0]
    h21 = -400*x[0]
    h22 = 200
    return np.array([[h11, h12], [h21, h22]])

# ==================== FUNÇÃO 3 ====================
def f3(x):
    """f3(x) = 0.1*(12 + x1^2 + (1+x2^2)/x1^2 + x1^2*x2^2 + 100/(x1^4*x2^4))"""
    return 0.1 * (12 + x[0]**2 + (1 + x[1]**2)/(x[0]**2) + 
                  x[0]**2 * x[1]**2 + 100/(x[0]**4 * x[1]**4))

def grad_f3(x):
    """Gradiente de f3"""
    x1, x2 = x[0], x[1]
    df1 = 0.1 * (2*x1 - 2*(1 + x2**2)/(x1**3) + 2*x1*x2**2 - 400/(x1**5 * x2**4))
    df2 = 0.1 * (2*x2/(x1**2) + 2*x1**2*x2 - 400/(x1**4 * x2**5))
    return np.array([df1, df2])

def hess_f3(x):
    """Hessiana de f3"""
    x1, x2 = x[0], x[1]
    h11 = 0.1 * (2 + 6*(1 + x2**2)/(x1**4) + 2*x2**2 + 2000/(x1**6 * x2**4))
    h12 = 0.1 * (-4*x2/(x1**3) + 4*x1*x2 + 1600/(x1**5 * x2**5))
    h21 = h12
    h22 = 0.1 * (2/(x1**2) + 2*x1**2 + 2000/(x1**4 * x2**6))
    return np.array([[h11, h12], [h21, h22]])

# ==================== FUNÇÃO 4 ====================
def f4(x):
    """f4(x) = (x1^2 + x2^2 + x1*x2)^2 + sin^2(x1) + cos^2(x2)"""
    return (x[0]**2 + x[1]**2 + x[0]*x[1])**2 + np.sin(x[0])**2 + np.cos(x[1])**2

def grad_f4(x):
    """Gradiente de f4"""
    q = x[0]**2 + x[1]**2 + x[0]*x[1]
    df1 = 2*q*(2*x[0] + x[1]) + 2*np.sin(x[0])*np.cos(x[0])
    df2 = 2*q*(2*x[1] + x[0]) - 2*np.cos(x[1])*np.sin(x[1])
    return np.array([df1, df2])

def hess_f4(x):
    """Hessiana de f4"""
    q = x[0]**2 + x[1]**2 + x[0]*x[1]
    dq1 = 2*x[0] + x[1]
    dq2 = 2*x[1] + x[0]
    
    h11 = 2*dq1**2 + 2*q*2 + 2*(np.cos(x[0])**2 - np.sin(x[0])**2)
    h12 = 2*dq1*dq2 + 2*q
    h21 = h12
    h22 = 2*dq2**2 + 2*q*2 - 2*(np.cos(x[1])**2 - np.sin(x[1])**2)
    return np.array([[h11, h12], [h21, h22]])

# ==================== FUNÇÃO 9 ====================
def f9(x):
    """f9(x) = 1.41*x1^4 - 12.76*x1^3 + 39.91*x1^2 - 51.93*x1 + 24.37 + (x2-3.9)^2"""
    return (1.41*x[0]**4 - 12.76*x[0]**3 + 39.91*x[0]**2 - 
            51.93*x[0] + 24.37 + (x[1] - 3.9)**2)

def grad_f9(x):
    """Gradiente de f9"""
    df1 = 5.64*x[0]**3 - 38.28*x[0]**2 + 79.82*x[0] - 51.93
    df2 = 2*(x[1] - 3.9)
    return np.array([df1, df2])

def hess_f9(x):
    """Hessiana de f9"""
    h11 = 16.92*x[0]**2 - 76.56*x[0] + 79.82
    h12 = 0
    h21 = 0
    h22 = 2
    return np.array([[h11, h12], [h21, h22]])

# ==================== CONFIGURAÇÃO DOS TESTES ====================
FUNCOES = {
    'f1': {
        'f': f1,
        'grad': grad_f1,
        'hess': hess_f1,
        'x_inicial': [np.array([0., 3.]), np.array([-1., -1.])],
        'x_otimo': np.array([2., 1.]),
        'nome_casos': ['caso_i', 'caso_ii']
    },
    'f2': {
        'f': f2,
        'grad': grad_f2,
        'hess': hess_f2,
        'x_inicial': [np.array([-5., 5.]), np.array([100., 1.])],
        'x_otimo': np.array([1., 1.]),
        'nome_casos': ['caso_i', 'caso_ii']
    },
    'f3': {
        'f': f3,
        'grad': grad_f3,
        'hess': hess_f3,
        'x_inicial': [np.array([0.5, 0.5]), np.array([3., -3.])],
        'x_otimo': np.array([1.743, 2.030]),
        'nome_casos': ['caso_i', 'caso_ii']
    },
    'f4': {
        'f': f4,
        'grad': grad_f4,
        'hess': hess_f4,
        'x_inicial': [np.array([3., 1.]), np.array([2., -2.])],
        'x_otimo': np.array([-0.1554, 0.6946]),
        'nome_casos': ['caso_i', 'caso_ii']
    },
    'f9': {
        'f': f9,
        'grad': grad_f9,
        'hess': hess_f9,
        'x_inicial': [np.array([0., 0.]), np.array([5., 5.])],
        'x_otimo': np.array([3.483, 3.9]),
        'nome_casos': ['caso_i', 'caso_ii']
    }
}
