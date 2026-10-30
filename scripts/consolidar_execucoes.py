"""Consolida repeticoes independentes do UCI HAR, sem fabricar observacoes.

Uso: python scripts/consolidar_execucoes.py --pattern 'results/uci_seed_*/metricas.csv'
"""
import argparse
import glob
from pathlib import Path
import pandas as pd

p=argparse.ArgumentParser()
p.add_argument('--pattern',default='results/uci_seed_*/metricas.csv')
p.add_argument('--out',default='results/resumo_repeticoes.csv')
args=p.parse_args()
paths=sorted(glob.glob(args.pattern))
if len(paths)<2:
    raise SystemExit('Sao necessarias no minimo duas execucoes independentes. Arquivos localizados: '+str(len(paths)))
frames=[]
for path in paths:
    df=pd.read_csv(path)
    if not set(['scenario','method','seed','accuracy','macro_f1']).issubset(df.columns):
        raise SystemExit('Colunas obrigatorias ausentes em '+path)
    frames.append(df)
all_df=pd.concat(frames,ignore_index=True)
if all_df.seed.nunique()<2:
    raise SystemExit('Todas as execucoes usam a mesma semente; nao sao repeticoes independentes.')
if all_df.source.ne('uci').any():
    raise SystemExit('Essa consolidacao e para dados UCI reais, nao use dados sinteticos como evidencia empirica.')
summary=(all_df.groupby(['scenario','method'],as_index=False)
    .agg(repeticoes=('seed','nunique'),acuracia_media=('accuracy','mean'),acuracia_dp=('accuracy','std'),
         macro_f1_media=('macro_f1','mean'),macro_f1_dp=('macro_f1','std')))
missing=summary.repeticoes.min()
if missing<2:
    raise SystemExit('Ao menos um cenario/metodo tem somente uma repeticao.')
Path(args.out).parent.mkdir(parents=True,exist_ok=True)
summary.to_csv(args.out,index=False)
print(summary.to_string(index=False))
print('Resumo salvo em',args.out)
