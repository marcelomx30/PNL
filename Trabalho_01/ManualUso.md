# Manual de Uso - Busca Linear (Grupo 04)

## Instalação

```bash
pip install numpy pandas
```

## Execução

### Executar todos os testes
```bash
python main.py
```

### Testar método específico
```bash
python metodo_secao_aurea.py    # Teste interno
python metodo_interpolacao.py   # Teste interno
python metodo_wolfe.py          # Teste interno
```

## Uso Programático

### Exemplo Básico
```python
from funcoes_teste import f1, grad_f1
from utils import criar_phi, criar_phi_derivada
from metodo_wolfe import condicoes_wolfe
import numpy as np

# Configurar
x_k = np.array([0., 3.])
d_k = -grad_f1(x_k)

# Criar φ(α)
phi = criar_phi(f1, x_k, d_k)
phi_derivada = criar_phi_derivada(f1, grad_f1, x_k, d_k)

# Executar
alpha, f_val, iters, tempo, conv = condicoes_wolfe(phi, phi_derivada)

print(f"α* = {alpha:.6f}")
print(f"Convergiu: {conv}")
```

### Testar Função Específica
```python
from main import executar_teste
from funcoes_teste import CASOS_TESTE

resultado = executar_teste('f1', CASOS_TESTE['f1'][0], 'wolfe')
print(resultado)
```

## Estrutura dos Arquivos

```
funcoes_teste.py       → Funções f1-f6 e gradientes
utils.py              → Criar φ(α) e φ'(α)
metodo_secao_aurea.py → Seção Áurea
metodo_interpolacao.py→ Interpolação
metodo_wolfe.py       → Wolfe
main.py               → Executa todos os testes
resultados.csv        → Resultados gerados
```

## Parâmetros

### Seção Áurea
```python
secao_aurea(phi, a=0, b=10, tol=1e-5, max_iter=1000)
```
- `a, b`: Intervalo inicial
- `tol`: Tolerância
- `max_iter`: Máximo de iterações

### Interpolação
```python
interpolacao(phi, phi_derivada, alpha0=1.0, tol=1e-5, max_iter=100)
```
- `alpha0`: Passo inicial
- `tol`: Tolerância
- `max_iter`: Máximo de iterações

### Wolfe
```python
condicoes_wolfe(phi, phi_derivada, alpha0=1.0, c1=1e-4, c2=0.9, 
                alpha_max=10.0, max_iter=100)
```
- `c1`: Constante de Armijo (0 < c1 < c2 < 1)
- `c2`: Constante de curvatura
- `alpha_max`: Passo máximo

## Resultados (resultados.csv)

| Coluna | Descrição |
|--------|-----------|
| funcao | Nome da função (f1-f6) |
| caso | caso_i ou caso_ii |
| metodo | secao_aurea, interpolacao, wolfe |
| f_inicial | Valor inicial |
| alpha_otimo | Passo ótimo |
| f_x_novo | Valor final |
| iteracoes | Número de iterações |
| tempo | Tempo de execução (s) |
| convergiu | True/False |

## Solução de Problemas

### Método não converge
- Aumentar `max_iter`
- Ajustar `tol`
- Verificar intervalo `[a,b]` (Seção Áurea)
- Testar outros `alpha0` (Interpolação/Wolfe)

### Direção não é de descida
```python
# Correto
d_k = -grad_f(x_k)

# Errado
d_k = grad_f(x_k)  # faltou o sinal negativo!
```

## Funções de Teste

- **f1**: (x₁-2)⁴ + (x₁-2x₂)²
- **f2**: Rosenbrock
- **f3**: Beale
- **f4**: Função composta 2D
- **f5**: Função 3D
- **f6**: Wood (4D)

## Conceitos

### Transformação Multidimensional → Unidimensional
```
f(x) multidimensional  →  φ(α) = f(x_k + α·d_k)  unidimensional
x_{k+1} = x_k + α*·d_k
```

### Direção de Descida
```
d_k = -∇f(x_k)  →  garante φ'(0) < 0
```

## Grupo 04 - UFC 2025
