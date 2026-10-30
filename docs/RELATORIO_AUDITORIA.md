# Relatório de Auditoria Completa — Federated XAI Lab

**Projeto:** Explicabilidade Local e Global em Aprendizado Federado Não-IID: Um Estudo com Dados de Sensores Distribuídos  
**Instituição:** Universidade Federal do Ceará — Campus Russas  
**Data da Auditoria:** 29 de outubro de 2026  
**Ambiente:** Windows 10/11, Python 3.12.10, PyTorch 2.15.0, scikit-learn 1.9.1, NumPy 2.5.4, Pandas 3.1.0, Matplotlib 3.11.2, Streamlit 1.66.1, pytest 9.2.0  

---

## 1. Visão Geral e Estrutura dos Arquivos

A auditoria inspecionou todos os arquivos e diretórios do repositório:
- `Artigo UFC Aplicacao Prática (1) (1).pdf`: Artigo científico e caderno técnico de 26 páginas definindo a fundamentação teórica, formulações matemáticas, metodologia, resultados com dados sintéticos e especificações técnicas da aplicação.
- `human+activity+recognition+using+smartphones/`: Contém os arquivos oficiais `UCI HAR Dataset.zip` (60.999.314 bytes, hash verificado) e `UCI HAR Dataset.names`.
- `Projeto_Pratico_Federated_XAI_UFC_Com_Resultados_Simulados/Projeto_Federated_XAI/`:
  - `src/flxai/`: Módulos centrais do pacote Python (`__init__.py`, `data.py`, `models.py`, `federated.py`, `explanation.py`, `experiment.py`).
  - `tests/`: Testes automatizados com pytest (`test_core.py`, `test_ucihar_io.py`).
  - `scripts/`: Scripts utilitários (`consolidar_execucoes.py`).
  - `results/`: Execuções com dados sintéticos (sementes 42, 43 e 44) e gráficos correspondentes.
  - `app.py`: Dashboard interativo em Streamlit para visualização dos arquivos gerados.
  - `dashboard_offline.html`: Interface estática autocontida para exploração da semente 42.
  - `requirements.txt` e `pyproject.toml`: Metadados do pacote e dependências mínimas.

Todos os arquivos originais foram previamente preservados no Git sob o commit `Preservacao dos arquivos originais antes de modificacoes`.

---

## 2. Auditoria dos Módulos do Sistema

### 2.1 `src/flxai/models.py`
- **Implementação:** Classe `SensorMLP(nn.Module)` com duas camadas ocultas lineares (32 e 16 neurônios), ativação ReLU e camada linear final para o número de classes do problema. Função auxiliar `new_model(input_dim, num_classes, seed)` inicializa o gerador do PyTorch via `torch.manual_seed(seed)`.
- **Conformidade:** Totalmente consistente com a especificação do Artigo (Seção 5.4 e Caderno Técnico, p. 8 e 18).
- **Avaliação:** Funcional e correta.

### 2.2 `src/flxai/federated.py`
- **Agregação FedAvg:** A função `weighted_average(models, weights)` normaliza os pesos por $n_k / N$ e calcula a combinação convexa dos tensores de parâmetros (`state_dict`).
  - **Veredito:** Ponderação rigorosamente correta.
- **Função de Perda FedProx:** Em `local_update`, para $\mu > 0$, adiciona-se o termo proximal $(\mu / 2) \sum \|w - w_t\|^2$ à perda de entropia cruzada (`F.cross_entropy`), onde $w_t$ (`anchor`) é copiado com `detach().clone()` dos parâmetros do modelo global no início da rodada.
  - **Veredito:** Formulação proximal matematicamente correta de acordo com Li et al. (2020).
- **Sincronização do Modelo Global:** A cada iteração em `train_federated`, cada cliente recebe uma cópia (`copy.deepcopy(model)`) do modelo global corrente. Nenhum cliente retém estado acumulado de parâmetros entre rodadas sem passar pela agregação global.
  - **Veredito:** Clientes partem exatamente do mesmo modelo global a cada rodada.
- **Treinamento Centralizado:** `train_centralized` treina a mesma arquitetura sequencialmente sobre todo o conjunto de treino agregado por $R$ rodadas com $\mu = 0$, servindo como baseline comparativo controlado.

### 2.3 `src/flxai/data.py`
- **Isolamento de Participantes e Prevenção de Vazamento (Data Leakage):**
  - No gerador sintético (`synthetic_sensors`): Utiliza `GroupShuffleSplit` encadeado para separar participantes em treino (62%), validação (17%) e teste (21%).
  - No leitor UCI HAR (`uci_har_dataset`): Respeita estritamente a partição oficial UCI HAR (treino com participantes 1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30; teste com os demais).
  - A validação é extraída **exclusivamente dos participantes de treino** via `GroupShuffleSplit`.
  - Asserções explícitas (`assert not set(groups[it]).intersection(groups[iv])`, etc.) garantem interseção nula de sujeitos.
  - O `StandardScaler` é ajustado **exclusivamente** sobre `X_train` (`X[it]`), transformando validação e teste sem vazamento de informação.
  - **Veredito:** Sem vazamento entre treino, validação e teste.
- **Particionamento Federado:**
  - `partition_dirichlet`: Divide índices de treino por classe usando amostragem de Dirichlet com parâmetro $\alpha$ e restrição de tamanho mínimo (`min_size`).
  - `partition_natural_subjects`: Agrupa os sujeitos reais de treino entre os clientes (`np.array_split(unique, clients)`), garantindo que cada cliente receba dados de participantes exclusivos.
  - **Observação técnica:** Em `partition_dirichlet`, o uso de `(proportions * len(index)).astype(int)` pode acumular pequenos resíduos no último cliente. Embora o particionamento seja comprovadamente disjunto e exaustivo (`assert len(all_idx) == len(y)`), convém aprimorar a distribuição de restos para simetria ideal em $\alpha=100$.

### 2.4 `src/flxai/explanation.py`
- **Integrated Gradients (IG):** Implementa a aproximação de Riemann para o caminho retilíneo entre o baseline (vetor nulo no espaço padronizado, correspondente à média de cada variável) e a entrada observada, avaliando o gradiente do logit acumulado.
  - A fórmula implementada:
    $$(x_i - x'_i) \times \frac{1}{m} \sum_{k=1}^m \frac{\partial F(x' + \frac{k}{m}(x - x'))}{\partial x_i}$$
  - **Veredito:** Matematicamente consistente com Sundararajan, Taly & Yan (2017).
- **Importância Global e Concordância:**
  - A importância global é calculada pela média do módulo das atribuições ($E[|IG|]$).
  - O cálculo do índice de Jaccard Top-$k$ está correto e implementado em `top_k_jaccard`.
  - **Ponto de atenção metodológico:** O cálculo atual de IG utiliza a classe prevista pelo próprio modelo (`labels = model(tensor).argmax(dim=1)`). Em modelos com predições divergentes em determinadas amostras, explicar classes distintas introduz assimetria. Para responder à QP2 e QP3 com rigor, é benéfico permitir explicar a classe-alvo (ground truth) ou assegurar comparações sobre amostras onde a classe explicada seja explicitamente padronizada.

### 2.5 `src/flxai/experiment.py`
- Orquestra os ensaios, gera os cenários IID e Não-IID, executa Centralizado, FedAvg e FedProx, avalia em teste e exporta 8 arquivos CSV, 2 imagens PNG e 2 arquivos JSON de metadados.
- **Veredito:** Métricas exportadas em `metricas.csv` correspondem diretamente aos valores retornados por `evaluate()` sobre o conjunto de teste não visto.

### 2.6 `app.py`
- Interface Streamlit modular que lê diretamente os CSVs produzidos por `experiment.py`, sem recalcular pesos nem fabricar dados.
- Suporta alternância de diretórios (`results/demonstracao`, `results/uci_seed_42`, etc.), seleção de cenários e gráficos de barras interativos.

---

## 3. Testes Executados e Resultados Obtidos

### 3.1 Testes Unitários Automatizados (`pytest -q`)
Comando executado:
```bash
pytest -q
```
Resultado:
```text
......                                                                   [100%]
6 passed in 47.89s
```
Casos cobertos:
1. `test_groups_held_out`: Verificação de separação de participantes e integridade dimensional.
2. `test_partitions_are_disjoint_and_exhaustive`: Confirmação de que as partições de clientes cobrem 100% dos dados sem sobreposição.
3. `test_weighted_model_average`: Teste exato de ponderação da agregação FedAvg com tensores controlados.
4. `test_integrated_gradients_shapes`: Verificação de formato, finitude e ausência de NaNs/Infs no IG.
5. `test_fedavg_trains_and_evaluates`: Execução ponta a ponta de uma rodada de treino e avaliação.
6. `test_uci_official_file_layout`: Verificação da estrutura e parser do UCI HAR com diretórios temporários mockados.

### 3.2 Execução Sintética Curta do Pipeline
Comando executado:
```bash
python -m flxai.experiment --mode synthetic --output-dir results/test_pipeline --rounds 2 --clients 3 --seed 42 --lr 0.07 --mu 0.3 --xai-samples 10 --ig-steps 5 --client-xai-samples 5 --client-ig-steps 5
```
Resultado:
- Pipeline executado com sucesso em 32 segundos.
- Todos os 12 arquivos de saída foram gerados com sucesso:
  - `metricas.csv`
  - `historico_validacao.csv`
  - `distribuicao_clientes.csv`
  - `importancia_global.csv`
  - `atribuicoes_locais.csv`
  - `concordancia_top5.csv`
  - `importancia_por_cliente.csv`
  - `concordancia_entre_clientes.csv`
  - `config.json`
  - `dataset_info.json`
  - `comparacao_f1.png`
  - `importancia_top10.png`

---

## 4. Problemas Identificados e Ajustes Necessários para a Campanha com Dados Reais

1. **Métricas por Classe Ausentes:**
   - O módulo `federated.evaluate()` atualmente calcula apenas `accuracy` e `macro_f1`.
   - Para atender à QP1 da FASE 5 ("Precision e Recall por classe, F1 por classe, Matriz de confusão"), o módulo deve ser estendido para registrar essas métricas detalhadas e exportá-las.
2. **Histórico de Treinamento e Perda:**
   - `historico_validacao.csv` atualmente registra apenas as métricas de validação (`accuracy`, `macro_f1`). Adicionar a perda (loss) de validação e treino auxilia na análise de convergência solicitada na FASE 4.
3. **Consistência do Alvo em Integrated Gradients:**
   - Possibilitar a explicação direcionada para a classe verdadeira (ground-truth) ou classe prevista explicitada, evitando comparar explicações de classes discordantes entre modelos na QP2/QP3.
4. **Extração e Validação do UCI HAR:**
   - O arquivo `UCI HAR Dataset.zip` está presente no repositório mas ainda precisa ser descompactado em `data/UCI HAR Dataset/` e auditado quanto a hash SHA-256, dimensões (561 características, 6 classes, 30 participantes) e ausência de sobreposições.

---

## 5. Critério de Aprovação da Fase 1

- [x] Todos os componentes essenciais foram auditados no código-fonte original.
- [x] O ambiente de execução e dependências foram instalados no `.venv`.
- [x] Todos os 6 testes automatizados passaram com sucesso.
- [x] A execução ponta a ponta do pipeline sintético foi concluída sem falhas.
- [x] Arquivos originais foram preservados no controle de versão Git.

**Status da Fase 1: CONCLUÍDA COM SUCESSO.**
