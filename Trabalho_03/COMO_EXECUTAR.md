# Como Executar - Trabalho 03

## ✅ Verificação Concluída

Todos os arquivos foram criados e testados com sucesso!

---

## 🚀 Comandos para Executar

### 1️⃣ Teste Rápido (RECOMENDADO - ~1 segundo)
```bash
cd ~/Documents/University/PNL/Trabalho_03
mise exec -- python teste_simples.py
```
**O que faz:** Testa apenas o Problema 1 com os 2 métodos

**Saída esperada:**
- Lagrangeana Aumentada: ✓ Convergiu
- Penalidade (β=10): ✓ Convergiu
- Tempo: ~0.04s

---

### 2️⃣ Teste Completo (48 testes - ~5-10 minutos)
```bash
cd ~/Documents/University/PNL/Trabalho_03
mise exec -- python main.py
```
**O que faz:**
- 4 problemas
- 3 casos por problema (pontos iniciais diferentes)
- 4 configurações: Lagrangeana + Penalidade (β=2, 10, 100)
- **Total: 48 testes**

**Arquivos gerados:**
- `resultados.csv` - Tabela completa com todos os resultados

---

## 📊 Estrutura dos Testes

```
Problema_1: 3 variáveis, restrição de igualdade
  ├─ caso_i:   x0 = [-2, 2, 0]
  ├─ caso_ii:  x0 = [5, 0, 1]
  └─ caso_iii: x0 = [-1.01, 0, 0.01]

Problema_2: 4 variáveis, 2 desigualdades + bounds (1 ≤ xi ≤ 5)
  ├─ caso_i:   x0 = [4, 4, 4, 4]
  ├─ caso_ii:  x0 = [5, 5, 5, 5]
  └─ caso_iii: x0 = [2, 3, 4, 5]

Problema_3: 2 variáveis, desigualdade circular
  ├─ caso_i:   x0 = [3, 2]
  ├─ caso_ii:  x0 = [0, 0]
  └─ caso_iii: x0 = [6, 5]

Problema_4: 2 variáveis, igualdade + desigualdade
  ├─ caso_i:   x0 = [-1, 0.5]
  ├─ caso_ii:  x0 = [-0.6, 0.25]
  └─ caso_iii: x0 = [-0.4, 0.4]
```

---

## 📈 Resultados Esperados

### ✅ Problemas que convergem bem:
- **Problema 1:** Ambos os métodos convergem em 2-3 iterações
- **Problema 3:** Convergência imediata (ponto inicial já está na solução)
- **Problema 2:** Converge com mais iterações

### ⚠️ Problemas desafiadores:
- **Problema 4:** Pode ter dificuldades de convergência devido à combinação de restrições de igualdade e desigualdade. Alguns casos podem não convergir completamente.

---

## 🔍 Interpretando a Saída

### Terminal:
```
Problema_1:
  caso_i: x0=[-2.  2.  0.]
    [ 1/48] [Lagrangeana     ] f*=9.273e-16 viol=3.98e-09 iters=  2 ✓
    [ 2/48] [Penalidade β=  2] f*=1.180e-15 viol=1.11e-08 iters=  2 ✓
    [ 3/48] [Penalidade β= 10] f*=1.180e-15 viol=1.11e-08 iters=  2 ✓
    [ 4/48] [Penalidade β=100] f*=1.180e-15 viol=1.11e-08 iters=  2 ✓
```

**Legenda:**
- `f*` = valor ótimo da função objetivo
- `viol` = violação total das restrições (quanto menor, melhor)
- `iters` = número de iterações externas
- `✓` = convergiu | `✗` = não convergiu

### Arquivo CSV:
Abre com Excel/LibreOffice e contém todas as métricas detalhadas.

---

## ⚙️ Se houver problemas

### Erro: "No module named 'numpy'"
```bash
cd ~/Documents/University/PNL/Trabalho_03
mise exec -- python -m pip install numpy pandas
```

### Erro: "mise not found"
```bash
# Usar Python direto (se numpy/pandas já instalados)
python main.py
# ou
python3 main.py
```

### Avisos de overflow no Problema 2
- **Normal!** Alguns pontos iniciais são ruins e causam overflow
- Os métodos detectam e param com segurança
- Verifique os casos que convergem no resultado final

---

## 📂 Arquivos do Projeto

```
Trabalho_03/
├── funcoes_teste.py                    # 4 problemas com restrições
├── metodo_penalidade.py                # Método de Penalidade
├── metodo_lagrangeana_aumentada.py     # Lagrangeana Aumentada
├── main.py                             # Programa completo (48 testes)
├── teste_simples.py                    # Teste rápido (recomendado)
├── MANUAL_DE_USO.md                    # Manual detalhado
├── README.md                           # Guia rápido
├── COMO_EXECUTAR.md                    # Este arquivo
└── resultados.csv                      # Gerado após execução
```

---

## 🎯 Resumo

**Para executar rapidamente:**
```bash
cd ~/Documents/University/PNL/Trabalho_03
mise exec -- python teste_simples.py
```

**Para gerar todos os resultados do trabalho:**
```bash
cd ~/Documents/University/PNL/Trabalho_03
mise exec -- python main.py
```

✅ Pronto para entrega!
