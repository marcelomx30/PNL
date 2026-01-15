# Trabalho 03 - Otimização Restrita
## Grupo 04 - Programação Não-Linear

### Métodos Implementados
1. **Método de Penalidade** (com taxas β = 2, 10, 100)
2. **Método de Lagrangeana Aumentada**

---

## Instalação Rápida

```bash
# Instalar dependências
pip install -r requirements.txt

# OU usando pacman (Arch Linux)
sudo pacman -S python-numpy python-pandas
```

---

## Execução

```bash
cd Trabalho_03
python main.py
```

---

## Estrutura do Projeto

```
Trabalho_03/
├── funcoes_teste.py                    # Problemas 1-4 com restrições
├── metodo_penalidade.py                # Método de Penalidade
├── metodo_lagrangeana_aumentada.py     # Método de Lagrangeana Aumentada
├── main.py                             # Programa principal
├── MANUAL_DE_USO.md                    # Manual completo
├── README.md                           # Este arquivo
├── requirements.txt                    # Dependências
└── resultados.csv                      # Resultados (gerado)
```

---

## Problemas Resolvidos

### Problema 1 (3 variáveis, restrição de igualdade)
- Função: `f1(x) = 0.01(x1-1)² + (x2-x1²)²`
- Restrição: `h(x) = x1 + x3² - 1 = 0`
- 3 pontos iniciais

### Problema 2 (4 variáveis, 2 desigualdades + bounds)
- Função: `f2(x) = x1*x4*(x1+x2+x3) + x3`
- Restrições: `g1(x) ≥ 0`, `g2(x) ≥ 0`, `1 ≤ xi ≤ 5`
- 3 pontos iniciais

### Problema 3 (2 variáveis, desigualdade)
- Função: `f3(x) = (x1²+x2-11)² + (x1+x2²-7)²`
- Restrição: `g(x) ≤ 0` (região circular)
- 3 pontos iniciais

### Problema 4 (2 variáveis, igualdade + desigualdade)
- Função: `f4(x) = (x1-2)² + (x2-1)²`
- Restrições: `g(x) ≤ 0`, `h(x) = 0`
- 3 pontos iniciais

**Total:** 4 problemas × 3 casos × 4 configurações (1 Lagrangeana + 3 Penalidades) = 48 testes

---

## Resultados Esperados

Após a execução, o programa gera:

1. **Saída no terminal:**
   - Progresso dos testes
   - Valores ótimos encontrados
   - Violação de restrições
   - Resumo comparativo

2. **Arquivo `resultados.csv`:**
   - Todos os resultados tabulados
   - Métricas de convergência
   - Violações de restrições
   - Tempos de execução

---

## Análise de Experimentos

### Método de Penalidade - Efeito de β

Conforme solicitado no projeto, testamos diferentes taxas de crescimento:

- **β = 2:** Crescimento lento, melhor condicionamento, mais iterações
- **β = 10:** Balanceado entre convergência e estabilidade
- **β = 100:** Convergência rápida, possível mal condicionamento

### Comparação: Penalidade vs Lagrangeana Aumentada

- **Lagrangeana:** Mais estável, não aumenta μ, atualiza multiplicadores
- **Penalidade:** Aumenta μ progressivamente, pode ter problemas numéricos

---

## Documentação Completa

Para informações detalhadas sobre:
- Uso programático
- Formato de entrada/saída
- Interpretação de resultados
- Fundamentos teóricos

Consulte: **[MANUAL_DE_USO.md](MANUAL_DE_USO.md)**

---

## Contato

**Grupo 04**
Universidade Federal do Ceará
Disciplina: Programação Não-Linear
Professor: Ricardo Coelho
Data de entrega: 16 de janeiro de 2026
