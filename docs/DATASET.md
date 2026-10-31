# Documentação do Dataset — UCI Human Activity Recognition Using Smartphones

**Projeto:** Explicabilidade Local e Global em Aprendizado Federado Não-IID: Um Estudo com Dados de Sensores Distribuídos  
**Instituição:** Universidade Federal do Ceará — Campus Russas  
**Autor:** Lucas de Souza Almeida (569990)  
**Data de Validação:** 29 de outubro de 2026  

---

## 1. Identificação e Proveniência

- **Nome Oficial:** Human Activity Recognition Using Smartphones
- **Repositório de Origem:** UCI Machine Learning Repository
- **URL Oficial:** [https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- **Citação:** Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013). *A public domain dataset for human activity recognition using smartphones*. In European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN).
- **Arquivo Interno de Origem:** `UCI HAR Dataset.zip` (60.999.314 bytes / 58,17 MB)
- **Hash SHA-256 Observado Localmente:** `2045e435c955214b38145fb5fa00776c72814f01b203fec405152dac7d5bfeb0` *(observado no arquivo obtido do repositório da UCI pelo autor; não confirmado em fonte externa independente)*
- **Diretório Local:** `data/UCI HAR Dataset/`

---

## 2. Descrição Geral dos Dados

O conjunto de dados reúne medições capturadas por sensores embarcados (acelerômetro triaxial e giroscópio triaxial) em um smartphone Samsung Galaxy S II fixado na cintura de 30 voluntários com idades entre 19 e 48 anos.

### 2.1 Características Extraídas
- **Total de características:** 561 variáveis numéricas contínuas por observação.
- **Domínio Temporal e Frequencial:**
  - Sinais temporais brutos do acelerômetro e giroscópio amostrados a 50 Hz.
  - Filtros passa-baixa e passa-alta aplicados para separar aceleração gravitacional e aceleração corporal.
  - Transformada Rápida de Fourier (FFT) aplicada para produzir componentes em frequência.
  - Medidas estatísticas extraídas: média (`mean()`), desvio padrão (`std()`), valor mediano (`mad()`), máximo (`max()`), mínimo (`min()`), energia por bandas (`bandsEnergy()`), entre outras.
- **Nomes Duplicados:** Foram identificados 42 nomes de características duplicados no arquivo oficial `features.txt` (por exemplo, `fBodyAcc-bandsEnergy()-1,8` aparece para os eixos X, Y e Z sem sufixo explícito). O pipeline preserva a indexação posicional de 1 a 561 para garantir rastreabilidade unívoca.

### 2.2 Classes de Atividades
O problema consiste em classificação multiclasse supervisionada em 6 categorias comportamentais:
1. `WALKING` (Caminhada em piso plano)
2. `WALKING_UPSTAIRS` (Subindo escadas)
3. `WALKING_DOWNSTAIRS` (Descendo escadas)
4. `SITTING` (Sentado)
5. `STANDING` (Em pé)
6. `LAYING` (Deitado)

---

## 3. Estrutura e Separação por Participantes

### 3.1 Participantes Totais: 30 Voluntários
- **Partição Oficial de Treinamento (70% dos participantes):**
  - 21 indivíduos: `[1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]`
  - Total de registros: 7.352 amostras.
- **Partição Oficial de Teste (30% dos participantes):**
  - 9 indivíduos: `[2, 4, 9, 10, 12, 13, 18, 20, 24]`
  - Total de registros: 2.947 amostras.
- **Sobreposição entre Treino e Teste:** Rigorosamente **0 participantes** em comum ($Train \cap Test = \emptyset$).

### 3.2 Subdivisão de Validação Interna
- Para garantir que a escolha e monitoramento de hiperparâmetros ocorram sem qualquer contaminação do conjunto de teste oficial, a validação é obtida exclusivamente por agrupamento de participantes de treinamento (`GroupShuffleSplit`, `test_size=0.20`, `random_state=seed`):
  - **Na semente de referência 42:** Validação com 5 participantes (`[1, 3, 15, 25, 27]`, 1.801 registros) e Treinamento Efetivo com 16 participantes (5.551 registros).
  - **Nas 5 sementes experimentais (42 a 46):** Treinamento Efetivo de $5.603 \pm 37$ registros (16 participantes) e Validação de $1.749 \pm 37$ registros (5 participantes), mantendo o teste oficial fixo em 2.947 registros (9 participantes).
  - **Sobreposição Treino/Validação/Teste:** Nenhuma sobreposição em nenhuma semente ($FinalTrain \cap Val = \emptyset$, $FinalTrain \cap Test = \emptyset$, $Val \cap Test = \emptyset$).

### 3.3 Distribuição de Classes por Partição

| Classe | Treinamento Oficial | Teste Oficial | Total |
|:---|:---:|:---:|:---:|
| WALKING | 1.226 | 496 | 1.722 |
| WALKING_UPSTAIRS | 1.073 | 471 | 1.544 |
| WALKING_DOWNSTAIRS | 986 | 420 | 1.406 |
| SITTING | 1.286 | 491 | 1.777 |
| STANDING | 1.374 | 532 | 1.906 |
| LAYING | 1.407 | 537 | 1.944 |
| **Total** | **7.352** | **2.947** | **10.299** |

---

## 4. Pré-processamento e Garantias de Não Vazamento

1. **Valores Ausentes:** Verificação exaustiva executada em todas as 10.299 linhas e 561 colunas. Não há valores ausentes (`NaN`), infinitos (`Inf`) ou nulos. Não foi necessária imputação ou geração de registros artificiais.
2. **Normalização / Padronização:**
   - O `StandardScaler` é ajustado **exclusivamente** sobre as amostras de treinamento efetivo da respectiva semente (5.551 amostras na semente 42; $5.603 \pm 37$ nas sementes 42–46).
   - Os conjuntos de validação e teste são apenas transformados com a média e desvio padrão do treinamento.
   - Verificação numérica:
     - Média dos atributos de treino após padronização: $\approx 8,3 \times 10^{-8}$ ($\approx 0$).
     - Desvio padrão dos atributos de treino: $\approx 1,0 \pm 9,3 \times 10^{-7}$.
3. **Limitação Metodológica:** O escalonamento centralizado de treino é utilizado no benchmark offline para prover equivalência de escala nas entradas da rede neural e nos baselines de Integrated Gradients, conforme documentado no artigo.

---

## 5. Arquivo de Validação Estruturado

Os dados brutos e verificações foram serializados no arquivo `results/validacao_dataset.json`, disponível para inspeção e testes automatizados.
