# Federated XAI Lab — UFC, Campus Russas

**Discentes:** JOAO PEDRO SCHONTZ DA SILVA (555036); LUCAS DE SOUZA ALMEIDA (569990)  
**Docente:** PITAGORAS GRAÇA MARTINS

Implementação de caráter acadêmico para comparar **treinamento centralizado, FedAvg e FedProx** em cenários **IID e Não-IID**, com explicações de previsões por **Integrated Gradients** e painel de resultados. Os clientes são **simulados no mesmo computador**. Não se deve inferir privacidade de implantação real nem fazer afirmações científicas a partir dos dados sintéticos.

## 1. Instalação

Requer Python 3.10 ou superior. No terminal, na pasta do projeto:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 2. Demonstração rápida, SEM dados reais

```bash
python -m flxai.experiment --mode synthetic --output-dir results/demonstracao --rounds 15 --clients 6 --seed 42 --lr 0.07 --mu 0.3 --xai-samples 50 --ig-steps 16
streamlit run app.py
```

Dados de treino/validação/teste são artificiais, mas possuem grupos de participantes separados e classes definidas. Os números são somente verificação da implementação; **não constituem resultados científicos em sensores reais**.

## 3. Experimentos com o UCI HAR real

Baixe o conjunto **Human Activity Recognition Using Smartphones** diretamente do repositório da UCI:
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

Extraia o diretório `UCI HAR Dataset/` em `data/` de modo que existam `data/UCI HAR Dataset/train/X_train.txt` e os demais arquivos oficiais.

```bash
python -m flxai.experiment --mode uci --dataset-path "data/UCI HAR Dataset" --output-dir results/uci_seed_42 --rounds 15 --clients 6 --xai-samples 120 --include-natural --seed 42
python -m flxai.experiment --mode uci --dataset-path "data/UCI HAR Dataset" --output-dir results/uci_seed_43 --rounds 15 --clients 6 --xai-samples 120 --include-natural --seed 43
python -m flxai.experiment --mode uci --dataset-path "data/UCI HAR Dataset" --output-dir results/uci_seed_44 --rounds 15 --clients 6 --xai-samples 120 --include-natural --seed 44
```

Para visualizar, no painel informe `results/uci_seed_42` na barra lateral.

Para consolidar media e desvio-padrao com no minimo duas sementes reais:

```bash
python scripts/consolidar_execucoes.py --pattern "results/uci_seed_*/metricas.csv" --out results/resumo_repeticoes.csv
```


**Importante:** a partição oficial de testes UCI HAR não participa do treino ou da escolha de hiperparâmetros; a validação é extraída apenas dos participantes do conjunto de treino. A normalização global do protótipo utiliza somente o treino agregado, porém isso é um *benchmark offline*, e NÃO um protocolo seguro de pré-processamento federado em implantação real.

## 4. Painel offline pronto para abrir

O arquivo `dashboard_offline.html` pode ser aberto diretamente em qualquer navegador,
sem instalar o Streamlit. Ele contem apenas a demonstracao com dados sinteticos ja executada,
e seus menus de cenario e metodo funcionam localmente. Nao carrega nem substitui os resultados reais do UCI HAR.

## 5. Testes automatizados

```bash
pip install pytest
pytest -q
```

## 6. Arquivos de saída

- `metricas.csv`: acurácia e macro-F1 no teste;
- `historico_validacao.csv`: métrica de validação por rodada;
- `distribuicao_clientes.csv`: quantidade de registros por classe e cliente;
- `importancia_global.csv`: média absoluta de Integrated Gradients por característica;
- `atribuicoes_locais.csv`: atribuições individuais para um cenário demonstrativo;
- `concordancia_top5.csv`: Jaccard entre rankings globais de métodos;
- `importancia_por_cliente.csv`: características relevantes calculadas em cada cliente;
- `concordancia_entre_clientes.csv`: Jaccard médio entre clientes do mesmo método, em dados locais de treinamento;
- `config.json`: parâmetros utilizados;
- `dataset_info.json`: origem e tamanho dos dados;
- `comparacao_f1.png`, `importancia_top10.png`: figuras exportadas.

## 7. Limites metodológicos

1. Clientes simulados sequencialmente, sem rede entre dispositivos físicos ou mecanismos de privacidade diferencial.
2. `StandardScaler` global do benchmark usa estatísticas agregadas da partição de treinamento; avaliar pré-processamento federado é trabalho futuro.
3. Integrated Gradients usa referência zero no espaço padronizado e atribuições sobre os logits das classes previstas; escolha da referência e quantidade de passos influenciam os resultados.
4. Cenários não-IID são criados com Dirichlet sobre classes; cenário natural opcional agrupa participantes reais do UCI HAR.
5. Centralizado e federado não possuem custo computacional equivalente. A comparação se restringe ao protocolo declarado.
6. Não utiliza SHAP, embora SHAP seja abordado na fundamentação bibliográfica. A técnica implementada é Integrated Gradients.
7. A análise de estabilidade e de fidelidade exige experimentos adicionais, incluindo repetições e perturbações controladas.

## 8. Estrutura

- `src/flxai/data.py`: ingestão e particionamento;
- `src/flxai/models.py`: MLP;
- `src/flxai/federated.py`: treino FedAvg/FedProx e avaliação;
- `src/flxai/explanation.py`: atribuições locais/globais;
- `src/flxai/experiment.py`: CLI, comparação e exportação;
- `app.py`: Streamlit;
- `tests/`: testes automatizados.

## 9. Referências iniciais

- McMahan et al. (2017), FedAvg: https://proceedings.mlr.press/v54/mcmahan17a.html
- Li et al. (2020), FedProx: https://arxiv.org/abs/1812.06127
- Sundararajan, Taly e Yan (2017), Integrated Gradients: https://proceedings.mlr.press/v70/sundararajan17a.html
- Lundberg e Lee (2017), SHAP, fundamentação teórica: https://arxiv.org/abs/1705.07874
- UCI HAR: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

## 10. Resultados sintéticos reproduzidos com três sementes (42, 43 e 44)

Foram executadas três demonstrações independentes com as mesmas configurações de treinamento:
6 clientes, 15 rodadas, uma época local, taxa de aprendizagem 0,07, FedProx com μ = 0,3,
50 amostras para explicabilidade global e 16 passos de integração.
A semente **regenera o próprio dataset sintético**, além de afetar particionamentos e
inicialização. Assim, a dispersão resume diferentes demonstrações artificiais, e **não**
validação estatística sobre um dataset real fixo.

Arquivos disponíveis:

- `results/demonstracao/`: execução da semente 42;
- `results/demonstracao_seed_43/`: execução da semente 43;
- `results/demonstracao_seed_44/`: execução da semente 44;
- `results/metricas_sinteticas_3_sementes.csv`: observações individuais;
- `results/resumo_sintetico_3_sementes.csv`: média e desvio-padrão de Macro-F1 e acurácia;
- `results/resumo_concordancia_sintetica.csv`: concordância Top-5 média e desvio-padrão;
- `results/fig_media_macro_f1_3_sementes.png`: comparação de Macro-F1;
- `results/fig_heterogeneidade_6_clientes.png`: visualização das distribuições por cliente.

**Médias de Macro-F1 (n = 3; sintético):**

| Cenário | Centralizado | FedAvg | FedProx |
|---|---:|---:|---:|
| IID (α = 100) | 0,952 | 0,404 | 0,398 |
| Não-IID moderado (α = 0,5) | 0,952 | 0,614 | 0,605 |
| Não-IID severo (α = 0,15) | 0,952 | 0,727 | 0,706 |

Essas diferenças não constituem evidência de superioridade de qualquer método: o gerador,
o orçamento de treinamento e os hiperparâmetros da demonstração não foram projetados
para responder a uma hipótese causal sobre a heterogeneidade Não-IID.

O **documento acadêmico integrado** traz as tabelas detalhadas, as limitações e o
caderno técnico de implementação, com trechos de código e comandos de reprodução.
