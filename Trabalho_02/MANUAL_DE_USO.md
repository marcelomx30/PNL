# Manual de Uso - Otimização Irrestrita (Grupo 04)

## Instalação

```bash
pip install numpy pandas matplotlib scipy
```

## Estrutura do Projeto

```
PNL/
├── Trabalho_01/              # Projeto anterior (busca linear)
│   ├── metodo_secao_aurea.py
│   ├── metodo_wolfe.py
│   └── utils.py
│
└── Trabalho_02/              # Este projeto
    ├── funcoes_teste.py
    ├── metodo_gradiente.py
    ├── metodo_newton.py
    ├── metodo_bfgs.py
    ├── metodo_polak_ribiere.py
    ├── main.py
    └── MANUAL_USO.md
```

## Execução

### Executar todos os testes (40 experimentos)
```bash
cd Trabalho_02
python main.py
```

### Resultado
- Arquivo `resultados.csv` gerado
- 5 funções × 2 pontos iniciais × 4 métodos = 40 testes

## Uso Programático

### Exemplo 1: Método do Gradiente
```python
from funcoes_teste import FUNCOES
from metodo_gradiente import metodo_gradiente
import numpy as np

# Configurar
f = FUNCOES['f1']['f']
grad = FUNCOES['f1']['grad']
x0 = np.array([0., 3.])

# Executar
x_opt, f_opt, iters, tempo, conv, hist = metodo_gradiente(f, grad, x0)

print(f"x* = {x_opt}")
print(f"f(x*) = {f_opt}")
print(f"Convergiu: {conv}")
```

### Exemplo 2: Método de Newton
```python
from metodo_newton import metodo_newton

# Precisa da Hessiana
hess = FUNCOES['f1']['hess']

x_opt, f_opt, iters, tempo, conv, hist = metodo_newton(f, grad, hess, x0)
```

### Exemplo 3: BFGS
```python
from metodo_bfgs import metodo_bfgs

# Não precisa da Hessiana!
x_opt, f_opt, iters, tempo, conv, hist = metodo_bfgs(f, grad, x0)
```

### Exemplo 4: Polak-Ribière
```python
from metodo_polak_ribiere import metodo_polak_ribiere

x_opt, f_opt, iters, tempo, conv, hist = metodo_polak_ribiere(f, grad, x0)
```

## Parâmetros

Todos os métodos aceitam:

```python
metodo_X(f, grad_f, x0, 
         tol=1e-6,              # Tolerância ||∇f|| < tol
         max_iter=1000,         # Máximo de iterações
         busca_linear='armijo'  # 'armijo' ou 'secao_aurea'
)
```

**Newton também precisa:**
```python
metodo_newton(f, grad_f, hess_f, x0, ...)
```

## Retorno

Todos os métodos retornam:
```python
(x_otimo, f_otimo, iteracoes, tempo, convergiu, historico)
```

Onde:
- `x_otimo`: solução encontrada
- `f_otimo`: f(x_otimo)
- `iteracoes`: número de iterações
- `tempo`: tempo de execução (segundos)
- `convergiu`: True/False
- `historico`: dict com trajetória
  - `historico['x']`: lista de pontos
  - `historico['f']`: lista de valores
  - `historico['grad_norm']`: norma do gradiente

## Funções de Teste

| Função | Descrição | Pontos Iniciais | x* |
|--------|-----------|-----------------|-----|
| f1 | (x₁-2)⁴ + (x₁-2x₂)² | (0,3), (-1,-1) | (2, 1) |
| f2 | Rosenbrock | (-5,5), (100,1) | (1, 1) |
| f3 | Função complexa | (0.5,0.5), (3,-3) | (1.743, 2.030) |
| f4 | Com sen²/cos² | (3,1), (2,-2) | (-0.1554, 0.6946) |
| f9 | Polinômio 4º grau | (0,0), (5,5) | (3.483, 3.9) |

## Visualizar Convergência

```python
import matplotlib.pyplot as plt

# Executar método
x_opt, f_opt, iters, tempo, conv, hist = metodo_gradiente(f, grad, x0)

# Plotar f(x) vs iterações
plt.plot(hist['f'])
plt.xlabel('Iteração')
plt.ylabel('f(x)')
plt.yscale('log')
plt.title('Convergência do Gradiente')
plt.grid(True)
plt.show()
```

## Resultados (resultados.csv)

| Coluna | Descrição |
|--------|-----------|
| funcao | Nome da função (f1-f9) |
| caso | caso_i ou caso_ii |
| metodo | gradiente, newton, bfgs, polak_ribiere |
| x_otimo | Solução encontrada |
| f_otimo | Valor da função |
| erro_x | ||x* - x_teorico|| |
| iteracoes | Número de iterações |
| convergiu | True/False |

## Métodos Implementados

### 1. Gradiente (Steepest Descent)
- Mais simples
- Convergência linear
- Direção: d = -∇f

### 2. Newton
- Convergência quadrática
- Usa Hessiana exata
- Direção: d = -H⁻¹∇f

### 3. BFGS (Quase-Newton)
- Aproxima H⁻¹
- Não precisa calcular Hessiana
- Atualização de rank-2

### 4. Polak-Ribière (Gradiente Conjugado)
- Direções conjugadas
- β_PR = (∇f_{k+1}^T(∇f_{k+1} - ∇f_k))/||∇f_k||²
- Mais eficiente que gradiente

## Solução de Problemas

### Erro: ModuleNotFoundError
```bash
# Instalar dependências
pip install numpy pandas

# Verificar se Trabalho_01 existe
ls ../Trabalho_01/
```

### Método não converge
- Aumentar `max_iter`
- Diminuir `tol`
- Testar outro ponto inicial
- Trocar busca linear

### Hessiana singular (Newton)
O código automaticamente usa gradiente se a Hessiana for singular.

## Grupo 04 - UFC 2025
