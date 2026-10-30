"""CLI principal: gera tabelas, graficos, configuracao, explicacoes e metricas."""
import argparse
import json
from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from .data import synthetic_sensors, uci_har_dataset, partition_dirichlet, partition_natural_subjects
from .federated import train_federated, train_centralized, evaluate
from .explanation import summarize_explanations, top_k_jaccard


def run(args):
    torch.set_num_threads(min(torch.get_num_threads(),2))
    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)
    output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    data=synthetic_sensors(seed=args.seed) if args.mode=='synthetic' else uci_har_dataset(args.dataset_path,seed=args.seed)
    cfg=vars(args).copy()
    (output/'config.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding='utf-8')
    (output/'dataset_info.json').write_text(json.dumps({
        'mode':args.mode,'n_train':len(data.y_train),'n_val':len(data.y_val),
        'n_test':len(data.y_test),'n_features':len(data.features),'classes':data.label_names,
        'preprocessing':'StandardScaler ajustado com treino agregado: benchmark offline, nao protocolo federado privado',
        'warning':'Dados artificiais: não usar como evidencia sobre desempenho em sensores reais' if args.mode=='synthetic' else 'UCI HAR real; particionamento de clientes simulado'},indent=2,ensure_ascii=False),encoding='utf-8')
    scenarios=[('IID','dirichlet',100.0),('Nao-IID moderado','dirichlet',.5),('Nao-IID severo','dirichlet',.15)]
    if args.include_natural:
        scenarios.append(('Participantes naturais','natural',None))
    results=[]; trace=[]; distributions=[]; explanations=[]; locals_=[]; concordance=[]; by_client=[]; cross_clients=[]
    for s_index,(scenario,kind,alpha) in enumerate(scenarios):
        partitions=(partition_natural_subjects(data.group_train,args.clients,args.seed)
           if kind=='natural' else partition_dirichlet(data.y_train,args.clients,alpha,args.seed+s_index,min_size=8))
        for client_id,subset in enumerate(partitions):
            for class_index,class_name in enumerate(data.label_names):
                distributions.append({'scenario':scenario,'client':client_id+1,'class':class_name,
                    'count':int(np.sum(data.y_train[subset]==class_index))})
        ranks={}
        for method in ['Centralizado','FedAvg','FedProx']:
            if method=='Centralizado':
                model, history=train_centralized(data,args.rounds,args.local_epochs,args.lr,args.batch_size,args.seed)
            else:
                model, history=train_federated(data,partitions,method,args.rounds,
                    args.local_epochs,args.lr,args.batch_size,args.mu,args.seed)
            score=evaluate(model,data.X_test,data.y_test)
            results.append({'scenario':scenario,'method':method,**score,'seed':args.seed,
                'source':args.mode,'test_subjects_held_out':True})
            for event in history: trace.append({'scenario':scenario,'method':method,**event})
            ranks[method], local=summarize_explanations(model,data.X_test,data.features,
                max_samples=args.xai_samples,steps=args.ig_steps)
            for item in ranks[method]: explanations.append({'scenario':scenario,'method':method,**item})
            if scenario=='IID' and method=='FedAvg':
                locals_=[{'scenario':scenario,'method':method,**row} for row in local]
            if method in ('FedAvg','FedProx'):
                local_rankings=[]
                for client_id,subset in enumerate(partitions):
                    local_rank,_=summarize_explanations(model,data.X_train[subset],data.features,
                        max_samples=args.client_xai_samples,steps=args.client_ig_steps)
                    local_rankings.append(local_rank)
                    by_client.extend([{'scenario':scenario,'method':method,
                        'client':client_id+1,**item} for item in local_rank])
                pairs=[top_k_jaccard(local_rankings[i],local_rankings[j],
                       k=min(5,len(data.features))) for i in range(len(local_rankings))
                       for j in range(i+1,len(local_rankings))]
                cross_clients.append({'scenario':scenario,'method':method,
                    'mean_pairwise_top5_jaccard':float(np.mean(pairs)),
                    'description':'Explicações sobre dados locais de treino; análise descritiva, não validação fora da amostra'})
        concordance.append({'scenario':scenario,'comparison':'FedAvg vs Centralizado',
            'jaccard_top5':top_k_jaccard(ranks['FedAvg'],ranks['Centralizado'],k=min(5,len(data.features)))})
        concordance.append({'scenario':scenario,'comparison':'FedProx vs FedAvg',
            'jaccard_top5':top_k_jaccard(ranks['FedProx'],ranks['FedAvg'],k=min(5,len(data.features)))})
    df=pd.DataFrame(results)
    df.to_csv(output/'metricas.csv',index=False)
    pd.DataFrame(trace).to_csv(output/'historico_validacao.csv',index=False)
    pd.DataFrame(distributions).to_csv(output/'distribuicao_clientes.csv',index=False)
    pd.DataFrame(explanations).to_csv(output/'importancia_global.csv',index=False)
    pd.DataFrame(locals_).to_csv(output/'atribuicoes_locais.csv',index=False)
    pd.DataFrame(concordance).to_csv(output/'concordancia_top5.csv',index=False)
    pd.DataFrame(by_client).to_csv(output/'importancia_por_cliente.csv',index=False)
    pd.DataFrame(cross_clients).to_csv(output/'concordancia_entre_clientes.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,4.6))
    pv=df.pivot(index='scenario',columns='method',values='macro_f1')
    pv.plot.bar(ax=ax,ylim=(0,1),rot=12)
    ax.set_title('Macro-F1 no conjunto de teste (participantes nao vistos)')
    ax.set_xlabel('Cenario de distribuicao'); ax.set_ylabel('Macro-F1')
    fig.tight_layout();fig.savefig(output/'comparacao_f1.png',dpi=170);plt.close(fig)
    tops=pd.DataFrame(explanations)
    f=tops[(tops.scenario=='Nao-IID severo')&(tops.method=='FedAvg')].nsmallest(10,'rank').sort_values('rank',ascending=False)
    fig,ax=plt.subplots(figsize=(8.5,4.3))
    ax.barh(f.feature,f.mean_abs_attribution)
    ax.set_xlabel('Media absoluta das atribuicoes Integrated Gradients')
    ax.set_title('Importancia global - FedAvg / Nao-IID severo')
    fig.tight_layout();fig.savefig(output/'importancia_top10.png',dpi=170);plt.close(fig)
    print('\nRESULTADOS',args.mode,'| semente',args.seed)
    print(df.to_string(index=False))
    print('\nArquivos em:',output.resolve())
    return df


def make_parser():
    parser=argparse.ArgumentParser(description='Aprendizado Federado Não-IID + XAI em sensores')
    parser.add_argument('--mode',choices=['synthetic','uci'],default='synthetic')
    parser.add_argument('--dataset-path',default='data/UCI HAR Dataset')
    parser.add_argument('--output-dir',default='results/execucao')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--clients',type=int,default=6)
    parser.add_argument('--rounds',type=int,default=5)
    parser.add_argument('--local-epochs',type=int,default=1)
    parser.add_argument('--lr',type=float,default=.035)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--mu',type=float,default=.1)
    parser.add_argument('--xai-samples',type=int,default=40)
    parser.add_argument('--ig-steps',type=int,default=16)
    parser.add_argument('--client-xai-samples',type=int,default=15)
    parser.add_argument('--client-ig-steps',type=int,default=10)
    parser.add_argument('--include-natural',action='store_true')
    return parser

if __name__=='__main__':
    run(make_parser().parse_args())
