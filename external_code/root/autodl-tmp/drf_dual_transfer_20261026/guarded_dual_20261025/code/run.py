"""Source-only selection, exact component controls, then frozen target evaluation."""
import argparse
import copy
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, log_loss
from cache import Cached, Pixels, MEAN, STD, build_caches, source_identity
from model import SemanticHead, GuardedDual, ReliabilityGate, gate_features, balanced_video_bce
from prepare import digest, dump, prepare


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def rows_for(run, part):
    return json.loads((Path(run)/'manifests'/(part+'.json')).read_text())


def groups(rows):
    by={}
    for i,row in enumerate(rows):by.setdefault(row['video_id'],[]).append(i)
    return list(by.values())


def metrics(rows, scores):
    if len(scores)!=len(rows) or not np.isfinite(scores).all():raise ValueError('Prediction coverage/nonfinite')
    yy=[];pp=[]
    for ids in groups(rows):
        if len({rows[i]['label'] for i in ids})!=1:raise ValueError('Label collision')
        yy.append(rows[ids[0]]['label']);pp.append(float(np.mean(scores[ids])))
    return {'video_auc':float(roc_auc_score(yy,pp)), 'video_log_loss':float(log_loss(yy,pp,labels=[0,1])),
            'videos':len(yy),'frames':len(rows),'aggregation':'mean_frame_probability_v1'}


class Paired(Dataset):
    def __init__(self, clean, aug):self.clean,self.aug=clean,aug
    def __len__(self):return len(self.clean)
    def __getitem__(self,i):return self.clean[i],self.aug[i]


def loader_for(config, dataset, training=False, rows=None, epoch=0, seed=2027):
    sampler=None
    if training:
        # REAL gets 1/2 mass; each observed fake method shares the remaining 1/2.
        kinds={row['method'] for row in rows if row['label']==1}
        counts={k:sum(r['method']==k for r in rows) for k in {'real'}|kinds}
        weights=[(.5 if r['label']==0 else .5/len(kinds))/counts[r['method']] for r in rows]
        gen=torch.Generator().manual_seed(seed+10000*epoch)
        sampler=WeightedRandomSampler(weights,len(dataset),replacement=True,generator=gen)
    return DataLoader(dataset,batch_size=config['batch_size'],sampler=sampler,shuffle=False,
        num_workers=config['workers'],pin_memory=True)


@torch.inference_mode()
def predict_cached(model, config, run, part, identity, semantic=False):
    rows=rows_for(run,part);ds=Cached(Path(run)/'cache'/part,rows,identity,semantic_only=semantic)
    result={'s':[],'d':[],'r':[]}
    model.eval()
    for batch in loader_for(config,ds):
        tokens=batch['tokens'].cuda(non_blocking=True).float()
        if semantic:
            z=model(tokens).float();out={'semantic_logit':z,'delta':torch.zeros_like(z),'residual_logit':torch.zeros_like(z)}
        else:out=model(batch['rgb'].cuda(non_blocking=True),tokens)
        for key,field in [('s','semantic_logit'),('d','delta'),('r','residual_logit')]:
            result[key].extend(out.get(field,torch.zeros_like(out['semantic_logit'])).float().cpu().tolist())
    return {k:np.asarray(v,dtype=np.float32) for k,v in result.items()}


def save_predictions(path, rows, pred, gate=None):
    with Path(path).open('x',encoding='utf-8') as f:
        for i,row in enumerate(rows):
            out={k:row[k] for k in ['dataset_id','video_id','family_id','frame_index','image_path','label']}
            out.update({k:float(v[i]) for k,v in pred.items()})
            if gate is not None:out['gate']=float(gate[i])
            f.write(json.dumps(out,ensure_ascii=False,allow_nan=False)+'\n')


def train_expert(config, run, seed, name, identity):
    run=Path(run);out=run/'runs'/f'seed{seed}'/name
    if out.exists():raise FileExistsError(out)
    spec=dict(config['model']);spec.update(config['ablations'].get(name,{}))
    semantic=name=='Semantic'
    seed_all(seed);anchor=SemanticHead().cuda()
    if not semantic:
        sp=run/'runs'/f'seed{seed}'/'Semantic'
        chosen=json.loads((sp/'best.json').read_text())['checkpoint']
        anchor.load_state_dict(torch.load(sp/chosen,map_location='cpu',weights_only=True)['model'],strict=True)
        model=GuardedDual(anchor,spec).cuda()
    else:model=anchor
    out.mkdir(parents=True,exist_ok=False)
    dump(out/'SPEC.json',{'spec':spec,'semantic_only':semantic,'seed':seed,'identity':identity,
        'train_config':config,'source_manifest_sha256':digest(run/'manifests/fit.json'),
        'validation_manifest_sha256':digest(run/'manifests/source_val.json'),
        'semantic_checkpoint_sha256':None if semantic else digest(sp/chosen),
        'scope':'Source fit only. No target scores used in training/selection'})
    fit=rows_for(run,'fit')
    clean=Cached(run/'cache/fit',fit,identity,semantic_only=semantic)
    aug=Cached(run/'cache/fit_aug',fit,identity,semantic_only=semantic)
    ds=Paired(clean,aug)
    lr=config['semantic_lr'] if semantic else config['dual_lr']
    params=[p for p in model.parameters() if p.requires_grad]
    if not semantic and not spec['use_residual']:
        # Exact structural exit. Separate extra-budget semantic control is distinct.
        pred=predict_cached(model,config,run,'source_val',identity)
        score=metrics(rows_for(run,'source_val'),1/(1+np.exp(-(pred['s']+pred['d']))))
        torch.save({'model':model.state_dict(),'spec':spec,'identity':identity,'epoch':0,'metrics':score},out/'epoch_000.pt')
        dump(out/'best.json',{'checkpoint':'epoch_000.pt',**score})
        return
    if semantic:
        opt=torch.optim.AdamW(params,lr=lr,weight_decay=config['weight_decay'])
    else:
        filt=[model.front.delta] if model.front.delta.requires_grad else []
        filt_ids={id(p) for p in filt}
        other=[p for p in params if id(p) not in filt_ids]
        opt=torch.optim.AdamW([{'params':other,'lr':lr},{'params':filt,'lr':lr*config['filter_lr_multiplier']}],weight_decay=config['weight_decay'])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=config['epochs'])
    best=-math.inf
    # Preserve epoch zero: random residual correction starts exactly at the anchor.
    for epoch in range(config['epochs']+1):
        loss_total=0.;count=0;start=time.monotonic()
        if epoch:
            model.train()
            for clean_batch,aug_batch in loader_for(config,ds,True,fit,epoch,seed):
                opt.zero_grad(set_to_none=True);losses=[];outs=[]
                for batch in [clean_batch,aug_batch] if config['paired_augmentation'] else [clean_batch]:
                    y=batch['label'].cuda().float();t=batch['tokens'].cuda(non_blocking=True).float()
                    out={'logit':model(t)} if semantic else model(batch['rgb'].cuda(non_blocking=True),t)
                    loss=F.binary_cross_entropy_with_logits(out['logit'].float(),y)
                    if not semantic and 'residual_logit' in out:
                        loss=loss+config['residual_aux_weight']*F.binary_cross_entropy_with_logits(out['residual_logit'].float(),y)
                    losses.append(loss);outs.append(out)
                loss=torch.stack(losses).mean()
                if len(outs)==2 and not semantic:
                    # Semantic anchor is frozen; consistency acts on correction only.
                    loss=loss+config['consistency_weight']*F.mse_loss(outs[0]['delta'],outs[1]['delta'])
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                loss.backward();nn.utils.clip_grad_norm_(params,1.);opt.step()
                loss_total+=float(loss.detach())*len(y);count+=len(y)
            sched.step()
        pred=predict_cached(model,config,run,'source_val',identity,semantic=semantic)
        probs=1/(1+np.exp(-np.clip(pred['s']+pred['d'],-50,50)))
        score=metrics(rows_for(run,'source_val'),probs)
        checkpoint=f'epoch_{epoch:03d}.pt'
        torch.save({'model':model.state_dict(),'spec':spec,'identity':identity,'epoch':epoch,'metrics':score},out/checkpoint)
        if score['video_auc']>best:
            best=score['video_auc'];pointer={'checkpoint':checkpoint,**score}
            (out/'best.json').write_text(json.dumps(pointer,indent=2)+'\n')
        entry={'seed':seed,'group':name,'epoch':epoch,'loss':loss_total/max(count,1),
               **score,'seconds':time.monotonic()-start}
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
        print('EPOCH',json.dumps(entry),flush=True)


def load_dual(config, run, seed, name, identity):
    folder=Path(run)/'runs'/f'seed{seed}'/name
    path=folder/json.loads((folder/'best.json').read_text())['checkpoint']
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state['identity']!=identity:raise ValueError('Checkpoint identity mismatch')
    model=GuardedDual(SemanticHead(),state['spec']).cuda()
    model.load_state_dict(state['model'],strict=True)
    return model.eval(),path


def video_arrays(rows,pred):
    ix=groups(rows)
    if len({len(x) for x in ix})!=1:raise ValueError('Gate requires complete equal frame counts')
    s=torch.tensor(np.stack([pred['s'][i] for i in ix]),device='cuda')
    d=torch.tensor(np.stack([pred['d'][i] for i in ix]),device='cuda')
    r=torch.tensor(np.stack([pred['r'][i] for i in ix]),device='cuda')
    y=torch.tensor([rows[x[0]]['label'] for x in ix],device='cuda',dtype=torch.float32)
    return s,d,r,y,ix


def fit_gates(config, run, seed, name, identity):
    model,checkpoint=load_dual(config,run,seed,name,identity)
    folder=Path(run)/'runs'/f'seed{seed}'/name/'gates'
    folder.mkdir(exist_ok=False)
    data={}
    for part in ['gate_fit','gate_holdout','source_val']:
        pred=predict_cached(model,config,run,part,identity)
        data[part]=(rows_for(run,part),pred)
        save_predictions(folder/(part+'_expert_predictions.jsonl'),*data[part])
    fs,fd,fr,fy,_=video_arrays(*data['gate_fit']); hs,hd,hr,hy,_=video_arrays(*data['gate_holdout'])
    fx=gate_features(fs,fd,fr);hx=gate_features(hs,hd,hr)
    center=fx.mean(0);scale=fx.std(0,unbiased=False).clamp_min(.05)
    fx=(fx-center)/scale;hx=(hx-center)/scale
    for mode in ['global','sample']:
        seed_all(seed);gate=ReliabilityGate(mode).cuda()
        optimizer=torch.optim.AdamW(gate.parameters(),lr=config['gate_lr'],weight_decay=.01)
        best=math.inf;best_state=None;best_step=None
        # Epoch zero is included, chosen on isolated source TRAIN gate_holdout loss.
        for step in range(config['gate_steps']+1):
            if step:
                gate.train();optimizer.zero_grad(set_to_none=True)
                g=gate(fx)
                probability=(fs+g[:,None]*fd).sigmoid().mean(-1)
                loss=balanced_video_bce(probability,fy)
                loss.backward();optimizer.step()
            if step%10==0:
                gate.eval()
                with torch.no_grad():
                    hp=(hs+gate(hx)[:,None]*hd).sigmoid().mean(-1)
                    hl=float(balanced_video_bce(hp,hy))
                if hl<best:
                    best=hl;best_state=copy.deepcopy(gate.state_dict());best_step=step
        gate.load_state_dict(best_state)
        torch.save({'model':best_state,'mode':mode,'center':center.cpu(),'scale':scale.cpu(),
            'expert_sha256':digest(checkpoint),'identity':identity,'selected_step':best_step,
            'holdout_balanced_video_log_loss':best,
            'selection':'equal class mass video BCE; isolated source TRAIN families; no target labels'},folder/(mode+'.pt'))
    report={}
    for part,(rows,pred) in data.items():
        s,d,r,y,ix=video_arrays(rows,pred);x=gate_features(s,d,r)
        variants={'SemanticAnchor':torch.zeros_like(y),'NoGate':torch.ones_like(y)}
        for mode in ['global','sample']:
            state=torch.load(folder/(mode+'.pt'),map_location='cpu',weights_only=True)
            gate=ReliabilityGate(mode).cuda();gate.load_state_dict(state['model'])
            with torch.no_grad():variants[mode]=gate((x-center)/scale)
        # Fixed diagnostics only; never choose a shuffle based on scores.
        variants['SampleMean']=variants['sample'].mean().expand_as(y)
        gen=torch.Generator(device='cpu').manual_seed(seed+917)
        permutation=torch.randperm(len(y),generator=gen).to(y.device)
        variants['SampleShuffled']=variants['sample'][permutation]
        report[part]={}
        for variant,g in variants.items():
            frameprob=(s+g[:,None]*d).sigmoid().detach().cpu().numpy()
            scores=np.zeros(len(rows),np.float32);gf=np.zeros(len(rows),np.float32)
            for i,ids in enumerate(ix):scores[ids]=frameprob[i];gf[ids]=float(g[i].detach())
            report[part][variant]={**metrics(rows,scores),'gate_mean':float(g.mean().detach()),
                'gate_std':float(g.std(unbiased=False).detach()),
                'balanced_video_log_loss':float(balanced_video_bce(torch.tensor(frameprob.mean(-1),device=y.device),y))}
            if part=='source_val':
                p={**pred,'probability':scores}
                save_predictions(folder/(part+'_'+variant+'.jsonl'),rows,p,gf)
    dump(folder/'SOURCE_GATE_REPORT.json',report)
    print('GATE_REPORT',seed,name,json.dumps(report),flush=True)


def check_ablation_plan(config):
    full=config['model']
    for name,changes in config['ablations'].items():
        if name=='Full':
            if changes:raise ValueError('Full must have no overrides')
        elif len(changes)!=1 or len([k for k,v in changes.items() if k in full and v!=full[k]])!=1:
            raise ValueError('Ablation is not exactly one effective model field: '+name)
    if config['eval_frames']!=32 or config['train_frames']<2:
        raise ValueError('Unsupported frame protocol')


def gpu_lock():
    lock=Path('/root/autodl-tmp/.drf_guarded_gpu.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    probe=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                         capture_output=True,text=True,check=True)
    others=[int(x.strip()) for x in probe.stdout.splitlines() if x.strip().isdigit() and int(x.strip())!=os.getpid()]
    if others:
        raise RuntimeError('Existing GPU processes preserved; refusing overlapping run: '+str(others))
    return lock


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','cache','source','gates'])
    p.add_argument('config');p.add_argument('run');p.add_argument('--seed',type=int,default=2027)
    p.add_argument('--groups',nargs='+',default=['Semantic','Full'])
    a=p.parse_args();config=json.loads(Path(a.config).read_text());check_ablation_plan(config)
    run=Path(a.run)
    if a.command=='prepare':
        run.mkdir(parents=True,exist_ok=False)
        config['_config_path']=a.config
        prepare(config,run/'manifests');dump(run/'CONFIG.json',config)
        return
    if config!= {k:v for k,v in json.loads((run/'CONFIG.json').read_text()).items() if k!='_config_path'}:
        raise ValueError('Frozen run config differs')
    # One lock across all new runs; also reject pre-existing external GPU workloads.
    lock=gpu_lock()
    if a.command=='cache':build_caches(config,run);return
    identity=source_identity(config)
    if json.loads((run/'IDENTITY.json').read_text())!=identity:raise ValueError('Run identity differs')
    if a.command=='source':
        for name in a.groups:
            if name!='Semantic' and name not in config['ablations']:raise ValueError('Unknown group')
            train_expert(config,run,a.seed,name,identity)
    else:
        for name in a.groups:
            if name=='Semantic':continue
            fit_gates(config,run,a.seed,name,identity)


if __name__=='__main__':main()
