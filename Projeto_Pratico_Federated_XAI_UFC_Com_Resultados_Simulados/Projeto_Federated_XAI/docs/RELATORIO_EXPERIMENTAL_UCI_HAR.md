# Relatório Experimental — Federated XAI Lab (Dataset Real UCI HAR)

**Artigo Científico:** Explicabilidade Local e Global em Aprendizado Federado Não-IID: Um Estudo com Dados de Sensores Distribuídos  
**Instituição:** Universidade Federal do Ceará — Campus Russas  
**Data:** 29 de outubro de 2026  
**Autores da Implementação:** João Pedro Schontz da Silva, Lucas de Souza Almeida  
**Docente Responsável:** Pitágoras Graça Martins  

---

## 1. Descrição dos Dados Reais

- **Dataset:** UCI Human Activity Recognition Using Smartphones (UCI HAR).
- **Origem e Repositório:** UCI Machine Learning Repository (Anguita et al., 2013).
- **URL Oficial:** `https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones`
- **Hash SHA-256 do Arquivo Oficial:** `2045e435c955214b38145fb5fa00776c72814f01b203fec405152dac7d5bfeb0` (60.999.314 bytes).
- **Total de Amostras:** 10.299 registros (7.352 de treinamento oficial e 2.947 de teste oficial).
- **Participantes:** 30 indivíduos (idades entre 19 e 48 anos).
  - 21 participantes no treinamento oficial (`[1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]`).
  - 9 participantes no teste oficial (`[2, 4, 9, 10, 12, 13, 18, 20, 24]`).
- **Classes de Atividades (6 classes balanceadas):**
  1. `WALKING` (1.722 amostras totais)
  2. `WALKING_UPSTAIRS` (1.544 amostras totais)
  3. `WALKING_DOWNSTAIRS` (1.406 amostras totais)
  4. `SITTING` (1.777 amostras totais)
  5. `STANDING` (1.906 amostras totais)
  6. `LAYING` (1.944 amostras totais)
- **Características:** 561 variáveis contínuas derivadas de sinais temporais e frequenciais de acelerômetro e giroscópio triaxiais.
- **Integridade dos Dados:** 0 valores ausentes (`NaN`) ou infinitos (`Inf`).

---

## 2. Metodologia

### 2.1 Separação dos Dados e Prevenção de Vazamento (Data Leakage)
1. **Teste Intocável:** O conjunto de teste oficial (2.947 amostras dos participantes 2, 4, 9, 10, 12, 13, 18, 20 e 24) foi rigorosamente mantido fora de qualquer etapa de treinamento, ajuste ou partição federada.
2. **Subdivisão de Validação:** Utilizou-se `GroupShuffleSplit` (20% dos participantes de treinamento) exclusivamente sobre os participantes de treino. 5 participantes (`[1, 3, 15, 25, 27]`, totalizando 1.801 amostras) formaram a validação, restando 5.551 amostras de 16 participantes para treinamento efetivo dos clientes federados.
3. **Interseção Nula de Sujeitos:** Comprovada formalmente:
   $$S_{treino} \cap S_{val} = \emptyset, \quad S_{treino} \cap S_{teste} = \emptyset, \quad S_{val} \cap S_{teste} = \emptyset$$
4. **Padronização:** O `StandardScaler` foi ajustado exclusivamente no subconjunto de treinamento efetivo ($X_{treino}$, 5.551 amostras) e apenas aplicado nas partições de validação e teste.

### 2.2 Estratégias de Particionamento Federado (6 Clientes Simulados)
- **Cenário A — Aproximadamente IID:** Particionamento Dirichlet com $\alpha = 100$.
- **Cenário B — Não-IID moderado:** Particionamento Dirichlet com $\alpha = 0,5$.
- **Cenário C — Não-IID severo:** Particionamento Dirichlet com $\alpha = 0,15$.
- **Cenário D — Particionamento natural:** Agrupamento dos 16 participantes de treinamento entre os 6 clientes sem fragmentação de indivíduos entre clientes distintos.

### 2.3 Métricas Efetivas de Heterogeneidade dos Cenários

| Cenário | Mín. Amostras / Cliente | Máx. Amostras / Cliente | TVD Médio em Relação ao Global | Entropia Normalizada Média ($H$) | Classes / Cliente |
|:---|:---:|:---:|:---:|:---:|:---:|
| **IID ($\alpha = 100$)** | 897 | 954 | 0,039 | 0,992 | 6 a 6 |
| **Não-IID moderado ($\alpha = 0,5$)** | 375 | 1.565 | 0,519 | 0,591 | 5 a 6 |
| **Não-IID severo ($\alpha = 0,15$)** | 225 | 1.623 | 0,632 | 0,397 | 2 a 6 |
| **Natural (Participantes Reais)** | 669 | 1.121 | 0,036 | 0,993 | 6 a 6 |

---

## 3. Configuração Experimental

- **Hardware e SO:** Intel Core / Windows 10/11 x64, execução multi-thread CPU (2 threads ativas para determinismo de profiling).
- **Bibliotecas:** Python 3.12.10, PyTorch 2.15.0, scikit-learn 1.9.1, NumPy 2.5.4, Pandas 3.1.0, Matplotlib 3.11.2, SciPy 1.18.1.
- **Arquitetura da Rede:** Perceptron Multicamadas (MLP) com entrada de 561 dimensões, camada oculta 1 com 32 neurônios (ReLU), camada oculta 2 com 16 neurônios (ReLU), camada de saída com 6 neurônios lineares.
- **Hiperparâmetros Congelados:**
  - Otimizador: SGD puro.
  - Taxa de aprendizado ($\eta$): 0,07.
  - Tamanho de lote (batch size): 64.
  - Rodadas federadas ($R$): 15.
  - Épocas locais por rodada ($E$): 1.
  - Parâmetro proximal FedProx ($\mu$): 0,3.
  - Sementes independentes executadas: 42, 43, 44, 45 e 46 ($N = 5$).
- **Parâmetros de Explicabilidade (Integrated Gradients):**
  - Baseline: Vetor nulo no espaço padronizado ($\bar{x} = 0$).
  - Passos de integração: 16 passos de Riemann.
  - Verificação de estabilidade: Diferença relativa média entre 16 e 32 passos $< 0,04$ (aproximação estável).
  - Amostras de teste avaliadas: 120 amostras de teste mantidas fixas.

---

## 4. Resultados Experimentais Consolidados ($N = 5$ Sementes)

### 4.1 Desempenho no Teste Oficial (Participantes Não Vistos)

| Cenário de Distribuição | Método de Treinamento | Acurácia Média $\pm$ DP | Macro-F1 Média $\pm$ DP | Perda Média (Cross-Entropy) | Tempo Médio de Treino (s) |
|:---|:---|:---:|:---:|:---:|:---:|
| **IID ($\alpha = 100$)** | Centralizado | 0,9034 $\pm$ 0,0574 | 0,8955 $\pm$ 0,0728 | 0,4623 $\pm$ 0,4876 | 6,92 s |
| | FedAvg | **0,9166 $\pm$ 0,0158** | **0,9150 $\pm$ 0,0167** | 0,2008 $\pm$ 0,0292 | 2,30 s |
| | FedProx | 0,9100 $\pm$ 0,0224 | 0,9077 $\pm$ 0,0243 | 0,2164 $\pm$ 0,0406 | 2,57 s |
| **Não-IID moderado ($\alpha = 0,5$)** | Centralizado | 0,9034 $\pm$ 0,0574 | 0,8955 $\pm$ 0,0728 | 0,4623 $\pm$ 0,4876 | 1,27 s |
| | FedAvg | **0,9113 $\pm$ 0,0219** | **0,9089 $\pm$ 0,0250** | 0,2194 $\pm$ 0,0334 | 1,41 s |
| | FedProx | 0,9082 $\pm$ 0,0220 | 0,9055 $\pm$ 0,0249 | 0,2326 $\pm$ 0,0377 | 2,42 s |
| **Não-IID severo ($\alpha = 0,15$)** | Centralizado | **0,9034 $\pm$ 0,0574** | **0,8955 $\pm$ 0,0728** | 0,4623 $\pm$ 0,4876 | 1,33 s |
| | FedAvg | 0,8752 $\pm$ 0,0550 | 0,8637 $\pm$ 0,0739 | 0,2990 $\pm$ 0,0790 | 1,25 s |
| | FedProx | 0,8724 $\pm$ 0,0524 | 0,8613 $\pm$ 0,0705 | 0,3030 $\pm$ 0,0620 | 2,76 s |
| **Participantes naturais** | Centralizado | 0,9034 $\pm$ 0,0574 | 0,8955 $\pm$ 0,0728 | 0,4623 $\pm$ 0,4876 | 1,27 s |
| | FedAvg | **0,9281 $\pm$ 0,0092** | **0,9270 $\pm$ 0,0089** | 0,1784 $\pm$ 0,0160 | 1,33 s |
| | FedProx | 0,9245 $\pm$ 0,0110 | 0,9233 $\pm$ 0,0109 | 0,1880 $\pm$ 0,0176 | 3,11 s |

### 4.2 Resultados Individuais por Semente (Macro-F1 no Teste)

| Semente | Cenário | Centralizado | FedAvg | FedProx |
|:---:|:---|:---:|:---:|:---:|
| **42** | IID | 0,9420 | 0,8870 | 0,8663 |
| | Não-IID moderado | 0,9420 | 0,8645 | 0,8628 |
| | Não-IID severo | 0,9420 | 0,8780 | 0,8765 |
| | Natural | 0,9420 | 0,9290 | 0,9247 |
| **43** | IID | 0,7705 | 0,9318 | 0,9291 |
| | Não-IID moderado | 0,7705 | 0,9235 | 0,9286 |
| | Não-IID severo | 0,7705 | 0,8956 | 0,9009 |
| | Natural | 0,7705 | 0,9351 | 0,9336 |
| **44** | IID | 0,9266 | 0,9182 | 0,9195 |
| | Não-IID moderado | 0,9266 | 0,9169 | 0,9125 |
| | Não-IID severo | 0,9266 | 0,7333 | 0,7367 |
| | Natural | 0,9266 | 0,9200 | 0,9142 |
| **45** | IID | 0,9450 | 0,9170 | 0,9089 |
| | Não-IID moderado | 0,9450 | 0,9229 | 0,9133 |
| | Não-IID severo | 0,9450 | 0,9128 | 0,9038 |
| | Natural | 0,9450 | 0,9154 | 0,9101 |
| **46** | IID | 0,8933 | 0,9209 | 0,9149 |
| | Não-IID moderado | 0,8933 | 0,9164 | 0,9102 |
| | Não-IID severo | 0,8933 | 0,8986 | 0,8884 |
| | Natural | 0,8933 | 0,9352 | 0,9339 |

### 4.3 Testes Estatísticos Pareados (FedProx vs. FedAvg)

| Comparação / Cenário | $\Delta$ Médio (FedProx $-$ FedAvg) | $t$ Pareado | $p$-valor ($t$) | Estatística Wilcoxon | $p$-valor (Wilcoxon) | Significância ($\alpha = 0,05$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **IID** | $-0,0073 \pm 0,0083$ | $-1,956$ | 0,1221 | 1,0 | 0,1250 | Não significativo |
| **Não-IID moderado** | $-0,0034 \pm 0,0055$ | $-1,351$ | 0,2481 | 3,0 | 0,3125 | Não significativo |
| **Não-IID severo** | $-0,0024 \pm 0,0070$ | $-0,767$ | 0,4856 | 5,0 | 0,6250 | Não significativo |
| **Participantes naturais** | $-0,0037 \pm 0,0021$ | $-3,843$ | **0,0184** | 0,0 | 0,0625 | Significativo ($t$-test) |
| **Impacto Não-IID (IID vs Severo)** | $-0,0513 \pm 0,0757$ | $-1,515$ | 0,2043 | — | — | Não significativo |

### 4.4 Explicabilidade e Concordância dos Modelos (Integrated Gradients)

#### Concordância Top-5 entre Modelos (Índice de Jaccard Médio $\pm$ DP)
- **IID:**
  - FedAvg vs Centralizado: $0,669 \pm 0,336$
  - FedProx vs FedAvg: **$1,000 \pm 0,000$** (100% de sobreposição dos Top-5)
- **Não-IID moderado:**
  - FedAvg vs Centralizado: $0,440 \pm 0,148$
  - FedProx vs FedAvg: **$0,867 \pm 0,183$**
- **Não-IID severo:**
  - FedAvg vs Centralizado: $0,491 \pm 0,352$
  - FedProx vs FedAvg: **$0,933 \pm 0,149$**
- **Participantes naturais:**
  - FedAvg vs Centralizado: $0,667 \pm 0,000$
  - FedProx vs FedAvg: **$0,800 \pm 0,183$**

#### Concordância Top-5 Inter-Clientes (Sobre Dados Locais de Treino)

| Cenário de Distribuição | Jaccard Inter-Clientes (FedAvg) | Jaccard Inter-Clientes (FedProx) |
|:---|:---:|:---:|
| **IID** | **0,6068 $\pm$ 0,1768** | **0,6248 $\pm$ 0,1772** |
| **Não-IID moderado** | 0,1593 $\pm$ 0,0837 | 0,1710 $\pm$ 0,0948 |
| **Não-IID severo** | 0,1575 $\pm$ 0,0705 | 0,1629 $\pm$ 0,0723 |
| **Participantes naturais** | 0,3611 $\pm$ 0,1135 | 0,3790 $\pm$ 0,1107 |

---

## 5. Discussão dos Resultados

1. **Efeito da Heterogeneidade no Desempenho (QP1):**
   - No particionamento natural (Cenário D), os modelos federados apresentaram seu melhor desempenho (Macro-F1 médio de 0,9270 no FedAvg e 0,9233 no FedProx), superando inclusive a média do baseline Centralizado (0,8955), demonstrando que o particionamento que preserva a integridade de voluntários individuais produz gradientes locais ricos e representativos.
   - No particionamento sintético Dirichlet severo ($\alpha = 0,15$), a Macro-F1 média caiu para 0,8637 no FedAvg, com clientes apresentando extrema carência de classes (entropia normalizada de 0,397). Na semente 44, observou-se uma queda mais acentuada (Macro-F1 de 0,7333), demonstrando que o particionamento desbalanceado severo eleva a sensibilidade à inicialização dos pesos e à dinâmica estocástica do SGD.
2. **Comparação FedAvg vs. FedProx:**
   - Com o hiperparâmetro proximal inicial $\mu = 0,3$, o FedProx não demonstrou vantagem preditiva sobre o FedAvg em nenhum dos cenários Dirichlet sintéticos ($p > 0,12$). No cenário de participantes naturais, o FedAvg superou o FedProx por uma margem sutil de 0,37 pontos percentuais ($p = 0,018$).
   - O custo computacional do FedProx foi em média 1,5 a 2 vezes maior em tempo de CPU (devido ao cálculo contínuo do termo de penalização proximal $\frac{\mu}{2} \sum \|w - w_t\|^2$ a cada minibatch).
3. **Concordância Global entre Métodos (QP2):**
   - FedAvg e FedProx mantiveram concordância Top-5 de Integrated Gradients extremamente elevada em todos os cenários (Jaccard entre 0,800 e 1,000). As características mais importantes identificadas pelo modelo global de ambos os métodos são praticamente idênticas.
   - Em relação ao modelo Centralizado, a concordância é intermediária (0,44 a 0,67), refletindo trajetórias de otimização distintas decorrentes da descentralização.
4. **Concordância entre Clientes Locais (QP3):**
   - A heterogeneidade de classes afeta drasticamente a concordância das atribuições locais entre clientes.
   - No cenário IID, a concordância média entre pares de clientes locais foi de **0,607**, enquanto no cenário Não-IID moderado despencou para **0,159** e no Não-IID severo para **0,157** (queda de mais de 74%).
   - Esse resultado evidencia de forma irrefutável que as explicações locais calculadas sobre os dados de treino de cada cliente expressam o viés da distribuição de classes local daquele participante, e não o funcionamento do modelo global em todo o domínio do problema.

---

## 6. Limitações do Estudo

1. **Simulação em Software:** Os clientes foram executados sequencialmente em um único ambiente computacional. Não foram simuladas latência de rede, perdas de pacotes ou desconexões parciais de clientes.
2. **Padronização Centralizada de Benchmark:** Para permitir baseline equivalente e viabilizar o cálculo do Integrated Gradients a partir da referência nula média ($\bar{x} = 0$), os dados foram escalonados com os parâmetros de média e variância do conjunto agregado de treino. Em uma implantação federada estritamente privada, deve-se adotar padronização local ou protocolo seguro de agregação de médias.
3. **Hiperparâmetros Fixados:** A pesquisa adotou os parâmetros descritos no artigo original ($\eta = 0,07, \mu = 0,3, 15$ rodadas, batch 64). O valor de $\mu = 0,3$ pode ter sido conservador ou subótimo para o FedProx no UCI HAR, o que justifica investigações adicionais com variação controlada de $\mu \in [0.001, 1.0]$.
4. **Interpretação Não-Causal:** Atribuições de Integrated Gradients indicam a sensibilidade e relevância associativa das características na função aprendida pela rede neural, não implicando causalidade física entre as grandezas dos sensores.

---

## 7. Conclusões e Respostas às Questões de Pesquisa

### Resposta à QP1
A heterogeneidade estatística severa introduzida por desbalanceamento extremo de classes reduz o desempenho médio dos classificadores federados (Macro-F1 cai de 0,9150 em IID para 0,8637 em Não-IID severo) e eleva a dispersão entre repetições. Por outro lado, quando a distribuição respeita os agrupamentos naturais de participantes reais executando o protocolo completo de atividades, os modelos federados atingem desempenho superior a 0,927 de Macro-F1, mostrando alta capacidade de generalização para indivíduos não vistos no teste oficial.

### Resposta à QP2
As importâncias globais de características extraídas via Integrated Gradients apresentam concordância quase perfeita entre FedAvg e FedProx (Jaccard Top-5 médio entre 0,80 e 1,00). As divergências mais perceptíveis ocorrem entre os modelos federados e a referência centralizada (Jaccard entre 0,44 e 0,67), refletindo caminhos distintos na superfície de perda.

### Resposta à QP3
A concordância entre as características consideradas mais relevantes pelos clientes individuais é fortemente dependente do grau de heterogeneidade local. Em condições IID, os clientes concordam em mais de 60% dos seus atributos Top-5. Em cenários Não-IID moderados e severos, essa concordância desaba para menos de 16%, demonstrando empiricamente que a explicabilidade local calculada sobre os dados residentes nos clientes é dominada pela distribuição amostral local, tornando imperativo o uso de amostras de referência padronizadas para auditorias globais de modelos federados.
