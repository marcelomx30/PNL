# Manual de Uso - Trabalho 03
## Otimização Restrita - Grupo 04

### Métodos Implementados
- **Método de Penalidade**
- **Método de Lagrangeana Aumentada**

---

## 1. Requisitos

### Dependências
```bash
pip install numpy pandas
```

### Estrutura de Arquivos
```
Trabalho_03/
├── funcoes_teste.py                    # Funções objetivo e restrições
├── metodo_penalidade.py                # Implementação do Método de Penalidade
├── metodo_lagrangeana_aumentada.py     # Implementação da Lagrangeana Aumentada
├── main.py                             # Programa principal
├── MANUAL_DE_USO.md                    # Este arquivo
└── resultados.csv                      # Resultados (gerado após execução)
```

---

## 2. Execução do Programa

### Execução Completa
Para executar todos os testes (4 problemas × 3 casos × 2 métodos):

```bash
cd Trabalho_03
python main.py
```

### Saída Esperada
O programa irá:
1. Exibir o progresso dos testes no terminal
2. Mostrar resultados detalhados para cada problema e caso
3. Gerar arquivo `resultados.csv` com todos os resultados
4. Apresentar resumo comparativo dos métodos

---

## 3. Problemas Teste

### Problema 1
```
min f1(x) = 0.01(x1-1)² + (x2-x1²)²
s.a h(x) = x1 + x3² - 1 = 0

Pontos iniciais:
- caso_i:   x0 = [-2, 2, 0]
- caso_ii:  x0 = [5, 0, 1]
- caso_iii: x0 = [-1.01, 0, 0.01]
```

### Problema 2
```
min f2(x) = x1*x4*(x1 + x2 + x3) + x3
s.a g1(x) = 25 - x1*x2*x3*x4 ≥ 0
    g2(x) = 40 - (x1² + x2² + x3² + x4²) ≥ 0
    1 ≤ xi ≤ 5 (∀ i = 1,2,3,4)

Pontos iniciais:
- caso_i:   x0 = [4, 4, 4, 4]
- caso_ii:  x0 = [5, 5, 5, 5]
- caso_iii: x0 = [2, 3, 4, 5]
```

### Problema 3
```
min f3(x) = (x1² + x2 - 11)² + (x1 + x2² - 7)²
s.a g(x) = (x1-3)² + (x2-2)² - 25 ≤ 0

Pontos iniciais:
- caso_i:   x0 = [3, 2]
- caso_ii:  x0 = [0, 0]
- caso_iii: x0 = [6, 5]
```

### Problema 4
```
min f4(x) = (x1-2)² + (x2-1)²
s.a g(x) = 0.25*x1² + x2² - 1 ≤ 0
    h(x) = x1 - 2*x2 + 1 = 0

Pontos iniciais:
- caso_i:   x0 = [-1, 0.5]
- caso_ii:  x0 = [-0.6, 0.25]
- caso_iii: x0 = [-0.4, 0.4]
```

---

## 4. Parâmetros dos Métodos

### Método de Penalidade
- **β (beta)**: Taxa de crescimento do parâmetro de penalidade
  - Testado com β = 2, 10, 100
  - Valores maiores → convergência mais rápida, mas maior mal condicionamento

- **μ (mu)**: Parâmetro de penalidade inicial = 1.0
- **Atualização**: μ(k+1) = β × μ(k)

### Método de Lagrangeana Aumentada
- **μ (mu)**: Parâmetro de penalidade fixo = 10.0
- **λ (lambda)**: Multiplicadores de Lagrange (atualizados automaticamente)
- **Atualização dos multiplicadores**:
  - Igualdade: λi := λi - μ × hi(x)
  - Desigualdade: λj := max(0, λj - μ × gj(x))

---

## 5. Uso Programático

### Exemplo: Método de Penalidade
```python
from metodo_penalidade import metodo_penalidade
from funcoes_teste import PROBLEMAS
import numpy as np

# Selecionar problema
config = PROBLEMAS['Problema_1']
f = config['f']
grad_f = config['grad']
x0 = config['x_inicial'][0]  # Primeiro ponto inicial
h_eq = config['h_eq']
grad_h_eq = config['grad_h_eq']

# Executar otimização
x_opt, f_opt, iters, tempo, convergiu, historico = metodo_penalidade(
    f, grad_f, x0,
    h_eq=h_eq,
    grad_h_eq=grad_h_eq,
    beta=10,
    tol=1e-6,
    max_iter_ext=50
)

print(f"Solução ótima: {x_opt}")
print(f"Valor ótimo: {f_opt}")
print(f"Iterações: {iters}")
print(f"Convergiu: {convergiu}")
```

### Exemplo: Lagrangeana Aumentada
```python
from metodo_lagrangeana_aumentada import metodo_lagrangeana_aumentada
from funcoes_teste import PROBLEMAS
import numpy as np

# Selecionar problema
config = PROBLEMAS['Problema_4']
f = config['f']
grad_f = config['grad']
x0 = config['x_inicial'][0]
h_eq = config['h_eq']
grad_h_eq = config['grad_h_eq']
g_ineq = config['g_ineq']
grad_g_ineq = config['grad_g_ineq']

# Executar otimização
x_opt, f_opt, iters, tempo, convergiu, historico = metodo_lagrangeana_aumentada(
    f, grad_f, x0,
    h_eq=h_eq,
    grad_h_eq=grad_h_eq,
    g_ineq=g_ineq,
    grad_g_ineq=grad_g_ineq,
    mu=10.0,
    tol=1e-6,
    max_iter_ext=50
)

print(f"Solução ótima: {x_opt}")
print(f"Valor ótimo: {f_opt}")
print(f"Iterações: {iters}")
print(f"Convergiu: {convergiu}")
```

---

## 6. Formato dos Dados de Entrada

### Função Objetivo
```python
def f(x):
    """
    Parâmetros:
        x : numpy.ndarray - vetor de variáveis de decisão

    Retorna:
        float - valor da função objetivo
    """
    return valor
```

### Gradiente
```python
def grad_f(x):
    """
    Parâmetros:
        x : numpy.ndarray - vetor de variáveis de decisão

    Retorna:
        numpy.ndarray - vetor gradiente
    """
    return gradiente
```

### Restrições de Igualdade
```python
def h(x):
    """
    Formato: h(x) = 0

    Retorna:
        float - valor da restrição
    """
    return valor
```

### Restrições de Desigualdade
```python
def g(x):
    """
    Formato: g(x) ≥ 0

    Nota: Se a restrição original é g(x) ≤ 0, usar -g(x) ≥ 0

    Retorna:
        float - valor da restrição
    """
    return valor
```

### Restrições de Caixa
```python
bounds = (lower, upper)  # Tupla com limites inferior e superior
# Exemplo: bounds = (1, 5) significa 1 ≤ xi ≤ 5 para todo i
```

---

## 7. Formato dos Dados de Saída

### Retorno das Funções
```python
x_opt : numpy.ndarray
    Ponto ótimo encontrado

f_opt : float
    Valor da função objetivo no ponto ótimo

iters : int
    Número de iterações externas realizadas

tempo : float
    Tempo de execução em segundos

convergiu : bool
    True se o método convergiu, False caso contrário

historico : dict
    Dicionário contendo:
    - 'x': lista de pontos em cada iteração
    - 'f': valores da função objetivo
    - 'mu': valores do parâmetro de penalidade (ou penalização)
    - 'lambda_eq': multiplicadores de igualdade (só Lagrangeana)
    - 'lambda_ineq': multiplicadores de desigualdade (só Lagrangeana)
    - 'grad_norm': norma do gradiente
```

### Arquivo resultados.csv
Colunas:
- `problema`: Nome do problema (Problema_1, Problema_2, etc.)
- `caso`: Identificador do caso (caso_i, caso_ii, caso_iii)
- `metodo`: Método utilizado (penalidade, lagrangeana)
- `beta`: Taxa de crescimento β (apenas para Penalidade)
- `x0`: Ponto inicial
- `x_otimo`: Ponto ótimo encontrado
- `f_otimo`: Valor ótimo da função objetivo
- `f_inicial`: Valor inicial da função objetivo
- `reducao`: Redução no valor da função (f_inicial - f_otimo)
- `violacao_eq`: Violação total das restrições de igualdade
- `violacao_ineq`: Violação total das restrições de desigualdade
- `violacao_bounds`: Violação total das restrições de caixa
- `violacao_total`: Violação total de todas as restrições
- `iteracoes`: Número de iterações externas
- `tempo`: Tempo de execução (segundos)
- `convergiu`: Indicador de convergência (True/False)

---

## 8. Interpretação dos Resultados

### Convergência
- ✓ indica que o método convergiu (violação < tolerância)
- ✗ indica que não houve convergência no número máximo de iterações

### Violação de Restrições
- Valores próximos de zero indicam que as restrições foram satisfeitas
- Valores altos indicam infactibilidade ou convergência lenta

### Comparação de β (Método de Penalidade)
- β baixo (2): Convergência mais lenta, melhor condicionamento
- β médio (10): Balanço entre convergência e condicionamento
- β alto (100): Convergência rápida, possível mal condicionamento

### Comparação entre Métodos
- **Lagrangeana Aumentada**: Geralmente mais estável, não aumenta μ
- **Penalidade**: Pode convergir mais rápido, mas sujeito a mal condicionamento

---

## 9. Observações Importantes

### Transformação de Restrições
- Restrições de igualdade h(x) = 0 são usadas diretamente
- Restrições g(x) ≤ 0 devem ser transformadas em -g(x) ≥ 0
- Bounds são tratados automaticamente pelos métodos

### Pontos Iniciais
- O ponto inicial pode afetar significativamente a convergência
- Para problemas não-convexos, diferentes pontos podem levar a ótimos locais diferentes
- Os métodos tentam ajustar x0 para satisfazer bounds quando aplicável

### Limitações
- Ambos os métodos requerem que a função objetivo seja diferenciável
- A convergência não é garantida para problemas não-convexos
- Valores muito altos de μ podem causar problemas numéricos

---

## 10. Suporte e Contato

**Grupo 04 - Programação Não-Linear**
Universidade Federal do Ceará
Disciplina: Programação Não-Linear
Professor: Ricardo Coelho

Para dúvidas ou problemas, consulte o código fonte ou a documentação das funções.
