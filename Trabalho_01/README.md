Esse primeiro projeto consiste da implementa¸c˜ao dos algoritmos de busca linear.


Métodos Atribuídos ao Grupo 04:Segundo a tabela, o grupo 04 deve implementar:

Método da Seção Áurea (sem derivada)
Método da Interpolação (exato com derivada)
Condições de Wol# Métodos de Busca Linear - Grupo 04

**UFC - Ciência de Dados - Programação Não-Linear**

## 📁 Arquivos

```
funcoes_teste.py       → f1-f6 e gradientes
utils.py              → Criar φ(α) e φ'(α)  
metodo_secao_aurea.py → Seção Áurea (sem derivada)
metodo_interpolacao.py→ Interpolação (com derivada)
metodo_wolfe.py       → Wolfe (inexato)
main.py               → Executa 36 testes
MANUAL_USO.md         → Documentação completa
```

## 🚀 Uso

```bash
# Instalar
pip install numpy pandas

# Executar todos os testes
python main.py

# Saída: resultados.csv
```

## 📊 Resultados

**36 experimentos:** 6 funções × 2 casos × 3 métodos

| Método | Convergência | Iterações Médias |
|--------|--------------|------------------|
| Seção Áurea | 12/12 (100%) | 30.0 |
| Interpolação | 12/12 (100%) | 22.3 |
| Wolfe | 12/12 (100%) | 9.1 |

## 💻 Exemplo

```python
from funcoes_teste import f1, grad_f1
from utils import criar_phi, criar_phi_derivada
from metodo_wolfe import condicoes_wolfe
import numpy as np

x_k = np.array([0., 3.])
d_k = -grad_f1(x_k)

phi = criar_phi(f1, x_k, d_k)
phi_derivada = criar_phi_derivada(f1, grad_f1, x_k, d_k)

alpha, f_val, iters, tempo, conv = condicoes_wolfe(phi, phi_derivada)
print(f"α* = {alpha:.6f}, convergiu: {conv}")
```

## 📖 Documentação

Ver **MANUAL_USO.md** para detalhes.

---

**Grupo 04 © 2025**fe (inexato com derivada)

