import argparse
import copy
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

FEATURES = ['Gender','Tx','Age1b','Tier2_N','grp_rate','rcmistot',
            'gnrl_fid','TKPctCrt','NWFcls','NWFwrc','ORFwc',
            'SAwrS','SAsrS','SAtoS','RMwidRS','RMwdaRS']

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class NumericEncoder:
    def fit(self, frame, features, categorical):
        self.features = features
        self.cat_maps = {}
        for c in features:
            if c in categorical:
                non_null = frame[c].dropna().astype(str).unique().tolist()
                self.cat_maps[c] = {v: i for i,v in enumerate(sorted(non_null))}
            elif not pd.api.types.is_numeric_dtype(frame[c]):
                # Numeric-looking object columns are okay; reject genuine unlisted categories.
                s = pd.to_numeric(frame[c], errors='coerce')
                bad = frame[c].notna() & s.isna()
                if bad.any(): raise ValueError(f'{c} has nonnumeric categories; add it to --categorical')
        return self
    def transform(self, frame):
        out=[]
        for c in self.features:
            if c in self.cat_maps:
                v=frame[c].astype('string').map(self.cat_maps[c]).to_numpy(dtype=float, na_value=np.nan)
            else:
                v=pd.to_numeric(frame[c], errors='coerce').to_numpy(dtype=float)
            finite = np.isfinite(v)
            if np.any(v[finite] < 0):
                raise ValueError(f'{c}: negative observed inputs; -1 must represent missing only. Recode before training.')
            v[~finite] = -1.0
            out.append(v)
        return np.column_stack(out).astype('float32')

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU())
        self.head=nn.Linear(64,1)
    def forward(self,x): return self.head(self.encoder(x)).flatten()

def masked_copy(x, rate):
    observed=x.ne(-1.0)
    choose=(torch.rand_like(x)<rate) & observed
    return torch.where(choose,torch.full_like(x,-1.0),x)

def pretrain(model, x, epochs, lr, batch, mask_rate, device):
    if len(x)==0: raise ValueError('Pretraining set is empty')
    loader=DataLoader(TensorDataset(torch.from_numpy(x)),batch_size=batch,shuffle=True)
    optimizer=torch.optim.Adam(model.encoder.parameters(),lr=lr)
    history=[]
    for _ in range(epochs):
        model.train(); total=0.; n=0
        for (xb,) in loader:
            xb=xb.to(device); za=model.encoder(xb); zb=model.encoder(masked_copy(xb,mask_rate))
            loss=(1-nn.functional.cosine_similarity(za,zb,dim=1)).mean()
            optimizer.zero_grad();loss.backward();optimizer.step()
            total+=loss.item()*len(xb);n+=len(xb)
        history.append(total/max(n,1))
    return history

def fit_supervised(model, xtr,ytr,xv,yv, epochs,lr,batch,patience,device):
    loader=DataLoader(TensorDataset(torch.from_numpy(xtr),torch.from_numpy(ytr)),batch_size=batch,shuffle=True)
    criterion=nn.BCEWithLogitsLoss(); opt=torch.optim.Adam(model.parameters(),lr=lr)
    best_loss=float('inf');best_state=None;bad=0;hist=[]
    xv_t=torch.from_numpy(xv).to(device);yv_t=torch.from_numpy(yv).to(device)
    for epoch in range(epochs):
        model.train(); total=0.;n=0
        for xb,yb in loader:
            xb=xb.to(device);yb=yb.to(device)
            loss=criterion(model(xb),yb)
            opt.zero_grad();loss.backward();opt.step()
            total+=loss.item()*len(xb);n+=len(xb)
        model.eval()
        with torch.no_grad(): vl=criterion(model(xv_t),yv_t).item()
        hist.append({'epoch':epoch+1,'train_loss':total/n,'val_loss':vl})
        if vl<best_loss-1e-6:
            best_loss=vl;best_state=copy.deepcopy(model.state_dict());bad=0
        else:
            bad+=1
            if bad>=patience:break
    model.load_state_dict(best_state)
    return pd.DataFrame(hist)

def predict(model,x,device):
    model.eval(); arr=[]
    with torch.no_grad():
        for start in range(0,len(x),1024):
            z=torch.from_numpy(x[start:start+1024]).to(device)
            arr.append(torch.sigmoid(model(z)).cpu().numpy())
    return np.concatenate(arr)

def scores(y,p):
    if len(y)==0:return {'n':0,'accuracy':np.nan,'average_precision':np.nan,'roc_auc':np.nan}
    return {'n':int(len(y)), 'accuracy':float(accuracy_score(y,p>=.5)),
            'average_precision':float(average_precision_score(y,p)) if len(np.unique(y))==2 else np.nan,
            'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan}

def choose_inner(labelled, groups, seed):
    if len(np.unique(groups))<2: raise ValueError('Need >=2 distinct training groups for validation')
    ss=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed)
    a,b=next(ss.split(labelled,groups=groups))
    if len(np.unique(labelled.iloc[a]['_target']))<2 or len(np.unique(labelled.iloc[b]['_target']))<2:
        raise ValueError('Inner validation fold has one class; supply more balanced groups or change seed')
    return a,b

def run(args):
    seed_all(args.seed)
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.csv)
    required=args.features+[args.target,args.student_id,args.school_id]
    if args.fold_column:required.append(args.fold_column)
    for c in required:
        if c not in df.columns:raise ValueError(f'Missing required column {c}')
    if df[args.student_id].isna().any() or df[args.school_id].isna().any():raise ValueError('Missing student/school IDs')
    labelled_mask=df[args.target].notna()
    labels=set(pd.to_numeric(df.loc[labelled_mask,args.target],errors='raise').unique())
    if not labels.issubset({0,1}):raise ValueError('Target must be prepared binary 0/1 labels; score columns are NOT valid targets')
    labelled=df[labelled_mask].copy()
    labelled['_target']=labelled[args.target].astype('float32')
    if len(labelled)<20:raise ValueError('Too few labelled rows')
    group_col=args.school_id if args.split=='school' else args.student_id
    if args.fold_column:
        folds=sorted(labelled[args.fold_column].dropna().unique(),key=str)
        if labelled[args.fold_column].isna().any():raise ValueError('Some labeled rows lack a fold assignment')
        if len(folds)!=5:raise ValueError('Need exactly five outer fold values')
        split_iter=[(labelled.index[labelled[args.fold_column]!=fold].to_numpy(), labelled.index[labelled[args.fold_column]==fold].to_numpy(),str(fold)) for fold in folds]
    else:
        cv=GroupKFold(n_splits=5)
        split_iter=[(labelled.iloc[a].index.to_numpy(),labelled.iloc[b].index.to_numpy(),str(i+1)) for i,(a,b) in enumerate(cv.split(labelled,groups=labelled[group_col]))]
    metrics=[];predictions=[]
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    for tridx,teidx,fold in split_iter:
        outer_train=labelled.loc[tridx].copy(); outer_test=labelled.loc[teidx].copy()
        if set(outer_train[group_col])&set(outer_test[group_col]):raise ValueError('Group leakage in outer fold '+fold)
        ai,bi=choose_inner(outer_train,outer_train[group_col].to_numpy(),args.seed)
        inner_train=outer_train.iloc[ai]; val=outer_train.iloc[bi]
        if set(inner_train[group_col])&set(val[group_col]):raise ValueError('Inner group leakage')
        # Restrict unlabeled pretraining examples to groups observed in *inner training*.
        # This conservative inductive policy excludes validation/test groups entirely.
        allowed=set(inner_train[group_col])
        pre_df=df[df[group_col].isin(allowed)].copy()
        # If supplied fold column marks rows, prevent any unassigned unlabeled test rows from entering.
        if args.fold_column:
            pre_df=pre_df[pre_df[args.fold_column].notna() & (pre_df[args.fold_column].astype(str)!=fold)]
        encoder=NumericEncoder().fit(pre_df,args.features,args.categorical)
        xpre=encoder.transform(pre_df)
        xtr=encoder.transform(inner_train); xv=encoder.transform(val); xte=encoder.transform(outer_test)
        ytr=inner_train['_target'].to_numpy(dtype='float32');yv=val['_target'].to_numpy(dtype='float32');yte=outer_test['_target'].to_numpy(dtype='int32')
        for name in ('MLP','MaskMLP'):
            seed_all(args.seed)
            model=MLP(len(args.features)).to(device)
            ph=[]
            if name=='MaskMLP':ph=pretrain(model,xpre,args.pretrain_epochs,args.lr,args.batch,args.mask_rate,device)
            hist=fit_supervised(model,xtr,ytr,xv,yv,args.epochs,args.lr,args.batch,args.patience,device)
            pd.DataFrame({'pretrain_loss':ph}).to_csv(root/f'{args.target}_{args.split}_fold{fold}_{name}_pretrain.csv',index=False)
            hist.to_csv(root/f'{args.target}_{args.split}_fold{fold}_{name}_learning.csv',index=False)
            torch.save({'state_dict':model.state_dict(),'feature_names':args.features,'target':args.target},root/f'{args.target}_{args.split}_fold{fold}_{name}.pt')
            p=predict(model,xte,device)
            for group_name,sel in [('all',np.ones(len(yte),dtype=bool)),('intervention',outer_test[args.intervention_column].astype(str).to_numpy()==str(args.intervention_value))] if args.intervention_column else [('all',np.ones(len(yte),dtype=bool))]:
                rec={'target':args.target,'split':args.split,'fold':fold,'model':name,'subset':group_name,**scores(yte[sel],p[sel])}
                metrics.append(rec)
            for row_id,yy,pp in zip(outer_test.index,yte,p):
                predictions.append({'row_index':int(row_id),'fold':fold,'model':name,'target':args.target,'y_true':int(yy),'probability':float(pp)})
            fig,ax=plt.subplots(figsize=(6,4));ax.plot(hist.epoch,hist.train_loss,label='train');ax.plot(hist.epoch,hist.val_loss,label='validation');ax.legend();ax.set(xlabel='Epoch',ylabel='BCE loss',title=f'{name} / fold {fold}');fig.tight_layout();fig.savefig(root/f'{args.target}_{args.split}_fold{fold}_{name}_loss.png',dpi=130);plt.close(fig)
            print(f'fold {fold} {name}: AP={scores(yte,p)["average_precision"]:.4f}  train={len(inner_train)} pretrain={len(pre_df)} val={len(val)} test={len(outer_test)}',flush=True)
    m=pd.DataFrame(metrics);p=pd.DataFrame(predictions)
    m.to_csv(root/'fold_metrics.csv',index=False);p.to_csv(root/'predictions.csv',index=False)
    summary=m.groupby(['target','split','model','subset'],dropna=False)[['accuracy','average_precision','roc_auc']].agg(['mean','std'])
    summary.to_csv(root/'summary_mean_std.csv')
    for name,sub in p.groupby('model'):
        fig,ax=plt.subplots(figsize=(6,4)); precision,recall,_=precision_recall_curve(sub.y_true,sub.probability)
        ax.plot(recall,precision,label=name);ax.set(xlabel='Recall',ylabel='Precision',title='Pooled out-of-fold precision–recall');ax.legend();fig.tight_layout();fig.savefig(root/f'{name}_precision_recall.png',dpi=130);plt.close(fig)
    (root/'config.json').write_text(json.dumps(vars(args),indent=2))
    print('\nSaved:',root,'\n',summary)

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--csv',required=True,help='Shared CSV containing features, 0/1 labels and IDs')
    ap.add_argument('--target',required=True,help='Precomputed binary reading outcome; NEVER a raw post-test score')
    ap.add_argument('--student-id',required=True);ap.add_argument('--school-id',required=True)
    ap.add_argument('--split',choices=['student','school'],required=True)
    ap.add_argument('--fold-column',default=None,help='Optional teammate-assigned outer fold ID column (five distinct values); includes unlabeled rows for conservative pretraining')
    ap.add_argument('--features',default=','.join(FEATURES),help='Comma-separated baseline feature columns')
    ap.add_argument('--categorical',default='Gender,Tx',help='Comma-separated categorical columns encoded using train-only mappings')
    ap.add_argument('--intervention-column',default='Tx');ap.add_argument('--intervention-value',default='1')
    ap.add_argument('--output',default='member3_results');ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--pretrain-epochs',type=int,default=100);ap.add_argument('--patience',type=int,default=15)
    ap.add_argument('--lr',type=float,default=.001);ap.add_argument('--batch',type=int,default=64)
    ap.add_argument('--mask-rate',type=float,default=.25);ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--cpu',action='store_true')
    args=ap.parse_args();args.features=[x.strip() for x in args.features.split(',') if x.strip()]
    args.categorical=[x.strip() for x in args.categorical.split(',') if x.strip()]
    if args.intervention_column and args.intervention_column not in pd.read_csv(args.csv,nrows=0).columns:
        print('No intervention column, disabling subgroup evaluation');args.intervention_column=None
    run(args)
if __name__=='__main__':main()
