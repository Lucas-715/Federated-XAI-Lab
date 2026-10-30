# Federated XAI Lab

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Trabalho Científico:** *Explicabilidade Local e Global em Aprendizado Federado Não-IID: Um Estudo com Dados de Sensores Distribuídos*  
**Instituição:** Universidade Federal do Ceará — Campus Russas  
**Discentes:** João Pedro Schontz da Silva (555036); Lucas de Souza Almeida (569990)  
**Docente Responsável:** Pitágoras Graça Martins  

---

## 1. Visão Geral

O **Federated XAI Lab** é um ambiente de pesquisa experimental e reprodutível desenvolvido para investigar os impactos da heterogeneidade estatística (distribuições IID vs. Não-IID) sobre o **desempenho preditivo** e a **explicabilidade de modelos federados** utilizando o método axiomático **Integrated Gradients** (Sundararajan et al., 2017).

O projeto avalia e compara três estratégias de aprendizado:
1. **Treinamento Centralizado:** Modelo de referência treinado sobre todos os dados de treinamento agregados.
2. **FedAvg (Federated Averaging):** Agregação ponderada por tamanho amostral dos parâmetros locais (McMahan et al., 2017).
3. **FedProx:** Otimização federada com penalização proximal local ($\mu = 0,3$) para conter a divergência de modelos locais em redes heterogêneas (Li et al., 2020).

Os experimentos são conduzidos sobre o dataset real de sensores vestíveis **UCI Human Activity Recognition Using Smartphones** (Anguita et al., 2013), respeitando integralmente a separação de indivíduos: 9 voluntários de teste fixos (2.947 amostras para avaliação final não vista) e particionamento dos 21 voluntários de treinamento em 16 para treino local ($5.603 \pm 37$ amostras) e 5 para validação ($1.749 \pm 37$ amostras) via `GroupShuffleSplit` por semente.

---

## 2. Estrutura do Repositório

```text
Federated-XAI-Lab/
├── data/
│   └── UCI HAR Dataset/          # Dataset real oficial (7.352 treino, 2.947 teste)
├── src/
│   └── flxai/
│       ├── __init__.py
│       ├── data.py               # Ingestão oficial UCI HAR, particionamentos Dirichlet e Natural
│       ├── models.py             # Arquitetura MLP compartilhada (561 -> 32 -> 16 -> 6)
│       ├── federated.py          # Treinamento FedAvg, FedProx, baseline Centralizado e avaliacao
│       ├── explanation.py        # Integrated Gradients, Jaccard Top-K e analise de rankings
│       └── experiment.py         # Orquestracao, coleta de metricas e exportacao
├── tests/
│       ├── test_core.py          # Testes unitarios do nucleo
│       ├── test_ucihar_io.py     # Testes de parser e layout do UCI HAR
│       └── test_partitioning.py  # Testes de disjuncao, integridade e cobertura federada
├── scripts/
│       ├── analisar_cenarios.py           # Analise estatistica e visual da heterogeneidade (TVD, Entropia)
│       ├── analisar_experimentos_reais.py   # Consolidacao multissemente e testes pareados
│       ├── recalcular_xai_checkpoints.py   # Recalculo de XAI controlado sobre os 60 checkpoints
│       └── consolidar_execucoes.py
├── results/
│       ├── uci_seed_42/                   # Execucao completa semente 42 (checkpoints, XAI, metricas)
│       ├── uci_seed_43/                   # Execucao completa semente 43
│       ├── uci_seed_44/                   # Execucao completa semente 44
│       ├── uci_seed_45/                   # Execucao completa semente 45
│       ├── uci_seed_46/                   # Execucao completa semente 46
│       ├── resumo_experimentos_reais.csv  # Medias e desvios-padrao consolidados (N=5)
│       ├── resumo_concordancia_top5_real.csv # Concordancia Top-5 (classe prevista e controlada)
│       ├── resumo_concordancia_entre_clientes_real.csv # Jaccard inter-clientes
│       ├── teste_estatistico_pareado.json # Testes t pareados e Wilcoxon
│       └── fig_resumo_macro_f1_real.png   # Grafico publicavel com barras de erro
├── docs/
│       ├── RELATORIO_AUDITORIA.md         # Relatorio formal de auditoria
│       ├── DATASET.md                     # Documentacao detalhada e proveniencia do dataset
│       └── RELATORIO_EXPERIMENTAL_UCI_HAR.md  # Relatorio cientifico completo
├── app.py                                 # Dashboard cientifico interativo em Streamlit
├── requirements.txt                       # Dependencias de producao
├── pyproject.toml                         # Metadados e build do pacote
├── LICENSE                                # Licenca MIT oficial
└── README.md
```

---

## 3. Instalação e Configuração

Requer **Python 3.10 ou superior**.

```bash
# 1. Clonar o repositorio e entrar no diretorio
git clone https://github.com/usuario/Federated-XAI-Lab.git
cd Federated-XAI-Lab

# 2. Criar e ativar o ambiente virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Instalar dependencias e pacote em modo editavel
pip install -r requirements.txt
pip install -e .
pip install pytest
```

---

## 4. Reprodução dos Experimentos Reais (UCI HAR)

Para reproduzir os experimentos com o dataset oficial da UCI nas 5 sementes independentes (42 a 46):

```bash
# Semente 42 (120 amostras estratificadas: 20 por classe, cobrindo todos os 9 sujeitos de teste)
python -m flxai.experiment --mode uci --dataset-path "data/UCI HAR Dataset" --output-dir results/uci_seed_42 --rounds 15 --clients 6 --lr 0.07 --mu 0.3 --xai-samples 120 --ig-steps 16 --client-xai-samples 25 --client-ig-steps 12 --include-natural --seed 42

# Sementes 43 a 46 seguem a mesma sintaxe com --seed correspondente.
```

### Recalcular XAI sobre Checkpoints Existentes
Para recalcular Integrated Gradients diretamente sobre os 60 checkpoints `.pt` salvos (avaliando tanto classe prevista quanto classe verdadeira controlada):

```bash
python scripts/recalcular_xai_checkpoints.py
```

### Consolidação Estatística Multissemente
Execute o script de consolidação para gerar as tabelas de resumo, testes pareados e figuras com barras de erro:

```bash
python scripts/analisar_experimentos_reais.py
```

---

## 5. Resultados Científicos Consolidados ($N = 5$ Sementes)

### Desempenho no Teste Oficial (Indivíduos Não Vistos)

| Cenário de Distribuição | Centralizado (Macro-F1) | FedAvg (Macro-F1) | FedProx (Macro-F1) | FedProx vs FedAvg ($p$-valor $t$-pareado) |
|:---|:---:|:---:|:---:|:---:|
| **IID ($\alpha = 100$)** | 0,8955 $\pm$ 0,0728 | **0,9150 $\pm$ 0,0167** | 0,9077 $\pm$ 0,0243 | $p = 0,122$ (Não significativo) |
| **Não-IID moderado ($\alpha = 0,5$)** | 0,8955 $\pm$ 0,0728 | **0,9089 $\pm$ 0,0250** | 0,9055 $\pm$ 0,0249 | $p = 0,248$ (Não significativo) |
| **Não-IID severo ($\alpha = 0,15$)** | **0,8955 $\pm$ 0,0728** | 0,8637 $\pm$ 0,0739 | 0,8613 $\pm$ 0,0705 | $p = 0,486$ (Não significativo) |
| **Participantes naturais** | 0,8955 $\pm$ 0,0728 | **0,9270 $\pm$ 0,0089** | 0,9233 $\pm$ 0,0109 | $p_t = 0,018$ (Tendência empírica; não-sig. pós-Bonferroni $\alpha_{corr}=0,0125$) |

### Síntese de Explicabilidade (Integrated Gradients)
- **Amostragem Estratificada Rigorosa:** 120 instâncias de teste selecionadas exatamente com 20 amostras por classe de atividade (Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying), com todos os 9 sujeitos de teste representados (`amostragem_xai.json`).
- **Concordância Top-5 Controlada pela Classe Verdadeira:**
  - FedProx vs. FedAvg: Jaccard Top-5 de **$0,933$ em IID**, **$1,000$ em Não-IID Moderado**, **$0,867$ em Não-IID Severo** e **$0,933$ em Natural**.
  - FedAvg vs. Centralizado: Jaccard Top-5 de **$0,689 \pm 0,363$ em IID**, **$0,463 \pm 0,380$ em Moderado**, **$0,425 \pm 0,248$ em Severo** e **$0,752 \pm 0,246$ em Natural**.
  - O controle da classe verdadeira revelou que divergências de classificação em instâncias limítrofes constituíam um fator de confusão relevante nas comparações puramente descritivas, embora coexistam com variações estocásticas nas representações dos modelos.
- **Desambiguação de Atributos:** Todas as 561 posições de características possuem identificadores posicionalmente únicos (`idx 0` a `idx 560`), eliminando qualquer risco de colisão entre os 42 nomes triplicados do UCI HAR (como `tGravityAcc-energy()-X`, identificado exclusivamente como `idx 56`).
- **Concordância Inter-Clientes (Dados Locais de Treino):** Em cenário IID, a concordância média entre pares de clientes locais foi de **0,607 $\pm$ 0,177**. Sob heterogeneidade, a concordância média recua para **0,159 $\pm$ 0,084** (Moderado) e **0,157 $\pm$ 0,070** (Severo), refletindo empiricamente a especialização dos dados de treino em subconjuntos restritos de classes combinada com a forte assimetria no volume amostral entre clientes no particionamento Dirichlet severo (e.g. de 11 a 2.572 amostras na semente 43, onde a amostragem de até 25 instâncias locais amplia a variância nos clientes menores).
- **Validação Axiomática Multi-Passo:** Verificação quantitativa do Axioma da Completude ($\sum IG_i \approx F(x) - F(0)$) avaliando o conjunto completo das 120 amostras de teste estratificadas (todas as 6 classes de atividade representadas). O erro relativo médio de completude situou-se confortavelmente abaixo de 1% em todas as cinco sementes já a partir de 16 passos (média geral de **0,557%** a 16 passos, **0,382%** a 32 passos e **0,238%** a 64 passos, com 100% das sementes mantendo média abaixo do limiar de 1% em todas as escalas avaliadas).

---

## 6. Executando o Dashboard Interativo (Streamlit)

O dashboard interativo permite inspecionar cada execução, examinar matrizes de confusão, curvas de validação de perda/F1, distribuições de classes, gráficos de Integrated Gradients e a síntese multissemente com testes de hipótese.

```bash
streamlit run app.py
```

---

## 7. Testes Automatizados

O repositório possui suite completa de 24 testes automatizados (testes unitários, axiomáticos, de particionamento, de conformidade do UCI HAR e teste de integração ponta a ponta do pipeline). Na ausência do dataset bruto no diretório local, os testes que dependem do arquivo bruto efetuam `skip` gracioso.

```bash
pytest -v tests
```
Saída esperada:
```text
============================== 24 passed in ~30s ==============================
```

---

## 8. Licença e Créditos

Este projeto é disponibilizado sob a [Licença MIT](LICENSE).  
A base **UCI Human Activity Recognition Using Smartphones** é de domínio público acadêmico disponibilizada pela Universidade da Califórnia, Irvine (UCI) e Autores (Anguita et al., 2013).
