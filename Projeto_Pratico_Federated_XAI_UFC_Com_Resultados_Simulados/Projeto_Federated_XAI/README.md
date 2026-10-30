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

Os experimentos são conduzidos sobre o dataset real de sensores vestíveis **UCI Human Activity Recognition Using Smartphones** (Anguita et al., 2013), respeitando integralmente a separação oficial de indivíduos entre particionamento, treino, validação e teste.

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
│       ├── analisar_cenarios.py  # Analise estatistica e visual da heterogeneidade (TVD, Entropia)
│       ├── analisar_experimentos_reais.py  # Consolidacao multissemente e testes pareados
│       └── consolidar_execucoes.py
├── results/
│       ├── uci_seed_42/          # Execucao completa semente 42
│       ├── uci_seed_43/          # Execucao completa semente 43
│       ├── uci_seed_44/          # Execucao completa semente 44
│       ├── uci_seed_45/          # Execucao completa semente 45
│       ├── uci_seed_46/          # Execucao completa semente 46
│       ├── resumo_experimentos_reais.csv  # Medias e desvios-padrao consolidados (N=5)
│       ├── teste_estatistico_pareado.json # Testes t pareados e Wilcoxon
│       └── fig_resumo_macro_f1_real.png   # Grafico publicavel com barras de erro
├── docs/
│       ├── RELATORIO_AUDITORIA.md         # Relatorio formal de auditoria
│       ├── DATASET.md                     # Documentacao detalhada e proveniencia do dataset
│       └── RELATORIO_EXPERIMENTAL_UCI_HAR.md  # Relatorio cientifico completo
├── app.py                        # Dashboard cientifico interativo em Streamlit
├── requirements.txt              # Dependencias de producao
├── pyproject.toml                # Metadados e build do pacote
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
# Semente 42
python -m flxai.experiment --mode uci --dataset-path "data/UCI HAR Dataset" --output-dir results/uci_seed_42 --rounds 15 --clients 6 --lr 0.07 --mu 0.3 --xai-samples 120 --ig-steps 16 --client-xai-samples 25 --client-ig-steps 12 --include-natural --seed 42

# Semente 43
python -m flxai.experiment --mode uci --dataset-path "data/UCI HAR Dataset" --output-dir results/uci_seed_43 --rounds 15 --clients 6 --lr 0.07 --mu 0.3 --xai-samples 120 --ig-steps 16 --client-xai-samples 25 --client-ig-steps 12 --include-natural --seed 43

# Semente 44
python -m flxai.experiment --mode uci --dataset-path "data/UCI HAR Dataset" --output-dir results/uci_seed_44 --rounds 15 --clients 6 --lr 0.07 --mu 0.3 --xai-samples 120 --ig-steps 16 --client-xai-samples 25 --client-ig-steps 12 --include-natural --seed 44

# Semente 45
python -m flxai.experiment --mode uci --dataset-path "data/UCI HAR Dataset" --output-dir results/uci_seed_45 --rounds 15 --clients 6 --lr 0.07 --mu 0.3 --xai-samples 120 --ig-steps 16 --client-xai-samples 25 --client-ig-steps 12 --include-natural --seed 45

# Semente 46
python -m flxai.experiment --mode uci --dataset-path "data/UCI HAR Dataset" --output-dir results/uci_seed_46 --rounds 15 --clients 6 --lr 0.07 --mu 0.3 --xai-samples 120 --ig-steps 16 --client-xai-samples 25 --client-ig-steps 12 --include-natural --seed 46
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
| **Participantes naturais** | 0,8955 $\pm$ 0,0728 | **0,9270 $\pm$ 0,0089** | 0,9233 $\pm$ 0,0109 | $p = 0,018$ (FedAvg $>$ FedProx) |

### Síntese de Explicabilidade (Integrated Gradients)
- **Concordância Global FedAvg vs. FedProx:** O índice de Jaccard Top-5 entre as características mais importantes dos modelos globais foi de **$1,000$ em IID**, **$0,867$ em Moderado** e **$0,933$ em Severo**, atestando convergência na hierarquia de atributos globais.
- **Concordância Inter-Clientes (Sobre Dados Locais de Treino):** Em cenário IID, a concordância média entre pares de clientes locais foi de **0,607**. Sob heterogeneidade severa, a concordância desaba para **0,157** (queda de 74%), comprovando que explicações locais em dados de treino refletem a especialização da amostra local e não o modelo global.

---

## 6. Executando o Dashboard Interativo (Streamlit)

O dashboard interativo permite inspecionar cada execução, examinar matrizes de confusão, curvas de validação de perda/F1, distribuições de classes, gráficos de Integrated Gradients e a síntese multissemente com testes de hipótese.

```bash
streamlit run app.py
```

---

## 7. Testes Automatizados

O repositório possui suite completa de testes automatizados:
```bash
pytest -q
```
Saída esperada:
```text
...................                                                      [100%]
19 passed in 25.19s
```

---

## 8. Licença e Créditos

Este projeto é disponibilizado sob a Licença MIT.  
A base **UCI Human Activity Recognition Using Smartphones** é de domínio público acadêmico disponibilizada pela Universidade da Califórnia, Irvine (UCI) e Autores (Anguita et al., 2013).
