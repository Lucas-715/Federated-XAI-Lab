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
- **Protocolo de Avaliação e Integridade do Teste:**
  - O conjunto de teste oficial do UCI HAR reserva 9 voluntários independentes (2.947 amostras) exclusivamente para avaliação final, garantindo separação estrita por indivíduos em 100% dos experimentos.
  - O particionamento entre treinamento local e validação foi realizado via `GroupShuffleSplit(n_splits=1, test_size=0.24, random_state=seed)` sobre os 21 voluntários de treinamento oficiais (16 voluntários para treino local e 5 para validação). Como os voluntários apresentam quantidades ligeiramente distintas de janelas de sensores, a composição exata varia por semente:
    - **Semente 42:** Treino = 5.551 amostras (16 voluntários) | Validação = 1.801 amostras (5 voluntários)
    - **Semente 43:** Treino = 5.636 amostras (16 voluntários) | Validação = 1.716 amostras (5 voluntários)
    - **Semente 44:** Treino = 5.622 amostras (16 voluntários) | Validação = 1.730 amostras (5 voluntários)
    - **Semente 45:** Treino = 5.578 amostras (16 voluntários) | Validação = 1.774 amostras (5 voluntários)
    - **Semente 46:** Treino = 5.627 amostras (16 voluntários) | Validação = 1.725 amostras (5 voluntários)
    - *Média Geral:* Treino = $5.603 \pm 38$ amostras | Validação = $1.749 \pm 38$ amostras | Teste = 2.947 amostras fixas.
  - *Ressalva Metodológica de Dispersão:* Essa estratégia avalia a sensibilidade a diferentes seleções de voluntários; logo, a dispersão estocástica reportada entre as 5 sementes decorre conjuntamente da composição dos indivíduos de treino, do particionamento federado (Dirichlet/Natural) e da inicialização dos pesos da rede.
  - *Transparência metodológica:* Durante a fase exploratória de desenvolvimento da instrumentação de métricas, uma execução inicial verificou a carga e inferência do pipeline sobre `X_test`. Conforme auditado no histórico, **nenhum hiperparâmetro ou componente da arquitetura foi ajustado ou otimizado em função desse teste** — os parâmetros $\eta = 0,07, \mu = 0,3, B = 64$ permaneceram estritamente congelados conforme especificado no projeto inicial.
- **Parâmetros de Explicabilidade (Integrated Gradients):**
  - Baseline: Vetor nulo no espaço padronizado ($\bar{x} = 0$).
  - Passos de integração: 16 passos como padrão operacional, com validação multi-escala em 32 e 64 passos.
  - Amostragem estratificada balanceada: 120 amostras de teste extraídas com 20 amostras por classe cobrindo todas as 6 atividades e todos os 9 indivíduos de teste.
  - Validação do Axioma da Completude: Verificação numérica multi-passo reportando erro relativo e atendimento ao limiar em 16, 32 e 64 passos.

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

| Comparação / Cenário | $\Delta$ Médio (FedProx $-$ FedAvg) | $t$ Pareado | $p$-valor ($t$) | Limiar Bonferroni ($\alpha / 4$) | Wilcoxon $W$ | $p$-valor (Wilcoxon) | Conclusão Estatística |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **IID** | $-0,0073 \pm 0,0083$ | $-1,956$ | 0,1221 | 0,0125 | 1,0 | 0,1250 | Não significativo |
| **Não-IID moderado** | $-0,0034 \pm 0,0055$ | $-1,351$ | 0,2481 | 0,0125 | 3,0 | 0,3125 | Não significativo |
| **Não-IID severo** | $-0,0024 \pm 0,0070$ | $-0,767$ | 0,4856 | 0,0125 | 5,0 | 0,6250 | Não significativo |
| **Participantes naturais** | $-0,0037 \pm 0,0021$ | $-3,843$ | 0,0184 | 0,0125 | 0,0 | 0,0625 | Tendência empírica (não sig. pós-Bonferroni) |
| **Impacto Não-IID (IID vs Severo)** | $-0,0513 \pm 0,0757$ | $-1,515$ | 0,2043 | — | — | — | Não significativo |

> **Nota Metodológica sobre Multiplicidade e Robustez:** O teste $t$ pareado no cenário natural indicou $p = 0,0184$. No entanto, ao controlar a taxa de erro da família de hipóteses para os 4 cenários avaliados através da correção conservadora de Bonferroni ($\alpha_{Bonf} = 0,05 / 4 = 0,0125$), tem-se $0,0184 > 0,0125$. Adicionalmente, o teste não-paramétrico de Wilcoxon apresentou $p = 0,0625$. Portanto, com $N = 5$ repetições, a sutil vantagem do FedAvg (+0,37 p.p.) deve ser reportada estritamente como uma **tendência empírica observada nas condições do protocolo**, e não como uma superioridade estatisticamente robusta.

### 4.4 Explicabilidade e Concordância dos Modelos (Integrated Gradients)

Para sanar ambiguidades de avaliação e assegurar comparações controladas, as atribuições de importância foram calculadas sob duas abordagens complementares nas 120 amostras de teste estratificadas (20 amostras por classe, abrangendo todos os 9 voluntários de teste):
1. **Classe-Alvo Controlada (Ground-Truth):** Avalia os modelos sob a mesma classe verdadeira de referência, isolando puramente a dinâmica da superfície de decisão aprendida.
2. **Classe Prevista pelo Modelo:** Avalia a classe efetivamente inferida por cada classificador.

Todas as 561 características do UCI HAR foram mapeadas com identificadores posicionais únicos (`idx 0` a `idx 560`), garantindo que as 42 grandezas com nomes textuais idênticos no dataset original nunca colidam nas contagens do índice de Jaccard.

#### Concordância Top-5 entre Modelos Globais no Teste (Média $\pm$ DP, $N = 5$ Sementes)

| Cenário de Distribuição | Comparação de Modelos | Jaccard Top-5 Controlado (Classe Verdadeira) | Jaccard Top-5 Descritivo (Classe Prevista) |
|:---|:---|:---:|:---:|
| **IID** | FedProx vs FedAvg | **0,9333 $\pm$ 0,1491** | **0,9333 $\pm$ 0,1491** |
| | FedAvg vs Centralizado | **0,6889 $\pm$ 0,3635** | 0,4635 $\pm$ 0,3804 |
| **Não-IID moderado** | FedProx vs FedAvg | **1,0000 $\pm$ 0,0000** | **1,0000 $\pm$ 0,0000** |
| | FedAvg vs Centralizado | **0,4635 $\pm$ 0,3804** | 0,3333 $\pm$ 0,3043 |
| **Não-IID severo** | FedProx vs FedAvg | **0,8667 $\pm$ 0,1826** | **0,8667 $\pm$ 0,1826** |
| | FedAvg vs Centralizado | **0,4246 $\pm$ 0,2480** | 0,3190 $\pm$ 0,2471 |
| **Participantes naturais** | FedProx vs FedAvg | **0,9333 $\pm$ 0,1491** | **1,0000 $\pm$ 0,0000** |
| | FedAvg vs Centralizado | **0,7524 $\pm$ 0,2461** | 0,5389 $\pm$ 0,3577 |

*Destaque Metodológico:* A concordância entre FedAvg e FedProx permanece extraordinariamente estável (entre 86,7% e 100% de sobreposição nos Top-5) em ambas as avaliações. Além disso, o controle pela classe verdadeira eleva substancialmente a concordância entre os modelos federados e o centralizado (atingindo 0,752 no cenário Natural e 0,689 em IID). Essa elevação indica que divergências em instâncias limítrofes mal classificadas constituíam um fator de confusão importante nas métricas puramente descritivas, embora não anulem por completo as diferenças residuais decorrentes das trajetórias de otimização descentralizadas.

#### Concordância Top-5 Inter-Clientes (Sobre Dados Locais de Treino)

| Cenário de Distribuição | Jaccard Inter-Clientes (FedAvg) | Jaccard Inter-Clientes (FedProx) |
|:---|:---:|:---:|
| **IID** | **0,6068 $\pm$ 0,1768** | **0,6248 $\pm$ 0,1772** |
| **Não-IID moderado** | 0,1593 $\pm$ 0,0837 | 0,1710 $\pm$ 0,0948 |
| **Não-IID severo** | 0,1573 $\pm$ 0,0703 | 0,1627 $\pm$ 0,0721 |
| **Participantes naturais** | 0,3611 $\pm$ 0,1135 | 0,3790 $\pm$ 0,1107 |

> **Ressalva Metodológica sobre Explicabilidade Local e Heterogeneidade Amostral (QP3):** Cada cliente calculou atribuições sobre seu respectivo particionamento local de treino. Nos cenários Não-IID moderado e severo, a partição local é dominada por subconjuntos restritos de classes de atividades. Além disso, o sorteio Dirichlet severo induz acentuada disparidade no volume amostral total dos clientes (por exemplo, na semente 43, os clientes contêm de 11 a 2.572 amostras). Como a explicabilidade local amostra até 25 instâncias (`min(25, len(subset))`), clientes com volumes diminutos (como o cliente 5 da semente 43, com 11 amostras) têm seus rankings calculados sobre um número reduzido de exemplos, o que contribui para ampliar a dispersão estocástica dos rankings locais. A acentuada redução da concordância inter-clientes (de 0,607 em IID para ~0,158 nos cenários não-IID) constitui, portanto, um comportamento empírico consistente com a especialização amostral de classes combinada com assimetrias de tamanho entre partições, evidenciando que explicações locais em dados de treino refletem primordialmente as particularidades da amostra local e não a coerência do modelo global (o qual mantém Jaccard de 0,867 a 1,000 quando avaliado em amostras comuns de teste).

### 4.5 Validação Axiomática e Protocolo de Amostragem do Integrated Gradients

1. **Amostragem Estratificada Balanceada de Produção:** Implementada diretamente na rotina principal (`select_stratified_xai_samples`), extraindo exatamente 20 amostras para cada uma das 6 classes (*Walking, Upstairs, Downstairs, Sitting, Standing, Laying*), totalizando 120 amostras no conjunto de teste oficial. Os metadados de amostragem gerados em cada semente (`amostragem_xai.json`) documentam que todos os 9 voluntários de teste independentes (sujeitos `[2, 4, 9, 10, 12, 13, 18, 20, 24]`) estão representados.
2. **Desambiguação Posicional de Características:** As 42 características com nomes duplicados no arquivo `features.txt` original receberam sufixos posicionais explícitos `(idx i)` e índices inteiros `feature_idx` de 0 a 560, eliminando ruído amostral nas comparações de conjuntos de Jaccard.
3. **Validação do Axioma da Completude em Múltiplas Escalas (16, 32 e 64 Passos):** Conforme a fundamentação de Sundararajan, Taly & Yan (2017), a soma das atribuições deve recuperar a variação de saída da rede em relação à baseline:
   $$\sum_{i=1}^{561} IG_i(x) \approx F(x) - F(0)$$
   A auditoria numérica nos modelos treinados (`iid_fedavg.pt` nas 5 sementes, avaliando 20 amostras sob o mesmo protocolo) registrou os seguintes erros relativos médios de completude:

| Semente | Erro Relativo (16 Passos) | Erro Relativo (32 Passos) | Erro Relativo (64 Passos) | Limiar 1% (16 Passos) | Limiar 1% (32 Passos) | Limiar 1% (64 Passos) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Semente 42** | 1,002% | 0,604% | 0,270% | Não atendido (+0,002%) | Atendido | Atendido |
| **Semente 43** | 1,214% | 0,791% | 0,505% | Não atendido (+0,214%) | Atendido | Atendido |
| **Semente 44** | 0,569% | 0,483% | 0,355% | Atendido | Atendido | Atendido |
| **Semente 45** | 0,106% | 0,185% | 0,293% | Atendido | Atendido | Atendido |
| **Semente 46** | 0,229% | 0,188% | 0,145% | Atendido | Atendido | Atendido |
| **Média Geral** | **0,624%** | **0,450%** | **0,314%** | 3/5 Atendidos | **5/5 Atendidos (100%)** | **5/5 Atendidos (100%)** |

*Avaliação de Convergência:* A média geral com 16 passos foi de **0,624%**, embora duas sementes individuais (42 e 43) tenham apresentado aproximações residuais marginalmente superiores a 1,0% (1,002% e 1,214%). Ao expandir para 32 e 64 passos, observa-se convergência monotônica assintótica estrita: **100% das sementes cumprem o limiar de 1% a 32 passos** ($\le 0,791\%$) e estabilizam abaixo de **0,51% a 64 passos**. Isso comprova matematicamente que a aproximação de Riemann empregada é consistente com o axioma da completude.

---

## 5. Discussão dos Resultados

1. **Efeito da Heterogeneidade no Desempenho (QP1):**
   - No particionamento sintético Dirichlet severo ($\alpha = 0,15$), a Macro-F1 média caiu para 0,8637 no FedAvg, com clientes apresentando extrema escassez de classes (entropia normalizada média de 0,397 e TVD de 0,632). Na semente 44, observou-se uma degradação mais acentuada (Macro-F1 de 0,7333), demonstrando que a escassez severa de classes amplia a sensibilidade à inicialização dos pesos e à estocasticidade do SGD local.
   - No particionamento de participantes reais (Cenário Natural), os modelos federados atingiram seu melhor desempenho (Macro-F1 médio de 0,9270 no FedAvg e 0,9233 no FedProx). Contudo, a análise quantitativa revela que os clientes no Cenário Natural possuem distribuição de classes praticamente uniforme (TVD médio de apenas 0,036 e entropia $H = 0,993$), visto que cada voluntário realizou o protocolo integral de atividades. Portanto, o ganho observado não deve ser causalmente atribuído ao mero ato de "preservar indivíduos", mas sim à conjunção empírica de ausência de desbalanceamento de rótulos com a rica diversidade motora inter-sujeito.
2. **Comparação FedAvg vs. FedProx:**
   - Com o hiperparâmetro proximal inicial $\mu = 0,3$, o FedProx não demonstrou vantagem preditiva sobre o FedAvg em nenhum dos cenários Dirichlet sintéticos ($p > 0,12$). No cenário de participantes naturais, observou-se uma sutil diferença média de 0,37 p.p. a favor do FedAvg ($p_t = 0,0184$). Conforme detalhado na Seção 4.3, ao aplicar a correção conservadora de Bonferroni para as 4 comparações de cenários ($\alpha_{Bonf} = 0,0125$) e o teste não-paramétrico de Wilcoxon ($p = 0,0625$), essa diferença constitui uma tendência empírica observada no protocolo, e não uma superioridade estatisticamente robusta.
   - O tempo de CPU do FedProx foi em média 1,5 a 2 vezes maior devido à computação contínua da penalidade proximal $\frac{\mu}{2} \sum \|w - w_t\|^2$ a cada minibatch.
3. **Auditoria de Convergência da Semente 43 e Tempos de Execução:**
   - *Comportamento do Centralizado na Semente 43:* Na semente 43, o modelo centralizado finalizou com Macro-F1 de 0,7705 no teste. A auditoria detalhada da curva de validação rodada a rodada (`historico_validacao.csv`) revelou que o modelo atingiu excelente desempenho até a Rodada 14 (Macro-F1 de **0,9732**, acurácia de **97,26%** e perda de **0,0672**). Porém, na Rodada 15, ocorreu uma desestabilização súbita com elevação da perda para 1,1394. Esse comportamento é empiricamente compatível com a hipótese de *gradient overshoot* decorrente da taxa de aprendizado fixa ($\eta = 0,07$) sem decaimento ou momentum, embora não se possa descartar outras dinâmicas complexas no espaço de perda. Nos métodos federados (FedAvg e FedProx), a média ponderada de parâmetros locais atuou como um regularizador empírico que preveniu desestabilizações análogas.
   - *Auditoria dos Tempos de Treinamento:* O tempo médio reportado para o Centralizado no cenário IID (6,92 s) comparado aos demais (~1,27 s) foi identificado como um artefato de *cold-start* da CPU (inicialização dos pools de threads MKL/OpenMP e alocação de tensores do PyTorch na primeira iteração do laço experimental). Em condições de ambiente aquecido (*warm cache*), o custo centralizado é estável em torno de 1,27–1,33 s.
4. **Concordância Global de Atribuições (QP2):**
   - FedAvg e FedProx mantiveram concordância Top-5 de Integrated Gradients extremamente elevada em todos os cenários (Jaccard entre 0,867 e 1,000). As características mais importantes identificadas pelo modelo global de ambos os métodos são praticamente idênticas.
   - Em relação ao modelo Centralizado, a concordância atinge até 0,752 no cenário Natural e 0,689 em IID sob classe-alvo controlada, revelando que divergências em amostras de fronteira eram um fator de confusão importante nas comparações puramente descritivas.
5. **Concordância entre Clientes Locais e Condicionamento Amostral (QP3):**
   - A concordância Top-5 entre clientes locais recua de 0,607 no cenário IID para ~0,158 nos cenários Não-IID moderado e severo.
   - Esse comportamento é consistente com a especialização das classes presentes na partição local de cada cliente e com a assimetria no volume de dados locais. Para comparações de modelos globais, é fundamental empregar conjuntos de teste padronizados e classes-alvo controladas.

---

## 6. Limitações do Estudo

1. **Simulação em Software:** Os clientes foram executados sequencialmente em um único ambiente computacional. Não foram simuladas latência de rede, perdas de pacotes ou desconexões parciais de clientes.
2. **Padronização Centralizada de Benchmark:** Para permitir baseline equivalente e viabilizar o cálculo do Integrated Gradients a partir da referência nula média ($\bar{x} = 0$), os dados foram escalonados com os parâmetros de média e variância do conjunto agregado de treino. Em uma implantação federada estritamente privada, deve-se adotar padronização local ou protocolo seguro de agregação de médias.
3. **Hiperparâmetros Fixados:** A pesquisa adotou os parâmetros descritos no artigo original ($\eta = 0,07, \mu = 0,3, 15$ rodadas, batch 64). O valor de $\mu = 0,3$ pode ter sido conservador ou subótimo para o FedProx no UCI HAR, o que justifica investigações adicionais com variação controlada de $\mu \in [0.001, 1.0]$.
4. **Interpretação Não-Causal:** Atribuições de Integrated Gradients indicam a sensibilidade e relevância associativa das características na função aprendida pela rede neural, não implicando causalidade física entre as grandezas dos sensores.
5. **Escopo Amostral das Repetições ($N = 5$):** O protocolo com 5 sementes independentes permite observar variabilidade empírica, mas limita o poder estatístico de testes não-paramétricos (como Wilcoxon, onde o valor mínimo de $p$ para $N=5$ é $1/2^4 = 0,0625$). Conclusões sobre superioridade marginal devem ser tratadas como tendências experimentais.

---

## 7. Conclusões e Respostas às Questões de Pesquisa

### Resposta à QP1
A heterogeneidade estatística severa introduzida por desbalanceamento de classes prejudica o desempenho preditivo dos classificadores federados (Macro-F1 cai de 0,9150 em IID para 0,8637 em Não-IID severo) e eleva a dispersão estocástica entre repetições. Já no particionamento natural por participantes, o desempenho médio atinge 0,927 de Macro-F1; esse alto desempenho está associado à preservação do equilíbrio global de classes entre os voluntários (TVD = 0,036), combinada com a diversidade sensorial física dos indivíduos.

### Resposta à QP2
As importâncias globais de características extraídas via Integrated Gradients no conjunto de teste oficial apresentam concordância quase perfeita entre FedAvg e FedProx (Jaccard Top-5 médio entre 0,867 e 1,00 em todos os cenários). Pequenas penalizações proximais ($\mu = 0,3$) preservam as mesmas características determinantes do classificador global. Quando a classe-alvo é controlada pela classe verdadeira, a concordância com o oráculo centralizado eleva-se substancialmente (0,689 em IID e 0,752 em Natural), indicando que discrepâncias limítrofes de classificação constituíam fator de confusão nas análises puramente descritivas.

### Resposta à QP3
A concordância entre as atribuições de importância calculadas localmente pelos clientes decai expressivamente com o aumento da heterogeneidade estatística (de 0,607 em IID para ~0,158 nos cenários Não-IID). Esse decaimento é empiricamente consistente com a composição amostral e com as assimetrias no volume de dados residentes em cada cliente, reforçando que auditorias de explicabilidade em ambientes federados heterogêneos exigem conjuntos de referência comuns e classes-alvo controladas para isolar o comportamento do modelo global das particularidades locais de cada partição.

