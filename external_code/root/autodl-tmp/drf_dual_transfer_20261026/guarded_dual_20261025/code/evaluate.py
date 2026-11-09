"""Freeze source-selected artifacts before one complete target evaluation."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import CLIPVisionModel
from cache import Pixels, MEAN, STD, source_identity
from model import ReliabilityGate, gate_features
from prepare import digest, dump
from run import load_dual, metrics, rows_for, video_arrays, gpu_lock


def freeze(config, run, seeds, names):
    run=Path(run);identity=source_identity(config);artifacts=[]
    for seed in seeds:
        for name in names:
            folder=run/'runs'/f'seed{seed}'/name
            ckpt=folder/json.loads((folder/'best.json').read_text())['checkpoint']
            for p in [ckpt,folder/'gates/sample.pt',folder/'gates/global.pt']:
                if not p.is_file():raise ValueError('Incomplete source run: '+str(p))
                artifacts.append({'path':str(p),'sha256':digest(p)})
    manifests={name:digest(run/'manifests'/(name+'.json')) for name in ['source_test',*config['targets']]}
    dump(run/'TARGET_EVAL_LOCK.json',{'identity':identity,'artifacts':artifacts,'manifests':manifests,
        'seeds':seeds,'groups':names,'selection':'source only; all declared targets, no per-target model choice',
        'all_requested_seeds_completed':sorted(seeds)==sorted(config['seeds'])})


@torch.inference_mode()
def evaluate(config, run):
    gpu_lease=gpu_lock()
    run=Path(run);lock=json.loads((run/'TARGET_EVAL_LOCK.json').read_text());identity=source_identity(config)
    if identity!=lock['identity']:raise ValueError('Frozen source code/backbone differs')
    for p in lock['artifacts']:
        if digest(p['path'])!=p['sha256']:raise ValueError('Frozen checkpoint changed')
    for name,h in lock['manifests'].items():
        if digest(run/'manifests'/(name+'.json'))!=h:raise ValueError('Frozen manifest changed')
    models={};gates={}
    for seed in lock['seeds']:
        for name in lock['groups']:
            key=f'{name}_seed{seed}'
            models[key],checkpoint=load_dual(config,run,seed,name,identity)
            gates[key]={}
            for mode in ['sample','global']:
                state=torch.load(checkpoint.parent/'gates'/(mode+'.pt'),map_location='cpu',weights_only=True)
                if state['expert_sha256']!=digest(checkpoint):raise ValueError('Gate/expert mismatch')
                gate=ReliabilityGate(mode).cuda();gate.load_state_dict(state['model']);gate.eval()
                gates[key][mode]=(gate,state['center'].cuda(),state['scale'].cuda())
    encoder=CLIPVisionModel.from_pretrained(config['clip_path'],local_files_only=True).cuda().eval().requires_grad_(False)
    mean=torch.tensor(MEAN,device='cuda')[None,:,None,None];std=torch.tensor(STD,device='cuda')[None,:,None,None]
    output=run/'target_evaluation';output.mkdir(exist_ok=False)
    summary={}
    for dataset in lock['manifests']:
        rows=rows_for(run,dataset)
        # Group rows in the manifest; no missing-frame skip or variable frame budget.
        collected={key:{'s':[],'d':[],'r':[]} for key in models}
        loader=DataLoader(Pixels(rows),batch_size=config['cache_batch_size'],num_workers=config['workers'],pin_memory=True)
        done=0
        for rgb in loader:
            rgb=rgb.cuda(non_blocking=True)
            with torch.autocast('cuda',dtype=torch.float16):
                tokens=encoder(pixel_values=(rgb-mean)/std).last_hidden_state
            # Match the exact float16 frozen-feature contract used for training.
            tokens=tokens.to(torch.float16).float()
            for key,model in models.items():
                out=model(rgb,tokens)
                for k,field in [('s','semantic_logit'),('d','delta'),('r','residual_logit')]:
                    collected[key][k].extend(out.get(field,torch.zeros_like(out['semantic_logit'])).cpu().tolist())
            done+=len(rgb)
            if done%4096<len(rgb) or done==len(rows):print('TARGET_PROGRESS',dataset,done,len(rows),flush=True)
        summary[dataset]={}
        for key,result in collected.items():
            pred={k:np.asarray(v,np.float32) for k,v in result.items()}
            s,d,r,y,ix=video_arrays(rows,pred)
            if s.shape[1]!=32:raise ValueError('Target is not exact32')
            x=gate_features(s,d,r)
            variants={'SemanticAnchor':torch.zeros_like(y),'NoGate':torch.ones_like(y)}
            for mode,(gate,center,scale) in gates[key].items():variants[mode]=gate((x-center)/scale)
            summary[dataset][key]={}
            probabilities={};weights={}
            for variant,g in variants.items():
                p=(s+g[:,None]*d).sigmoid().cpu().numpy();scores=np.zeros(len(rows),np.float32);gf=np.zeros(len(rows),np.float32)
                for i,ids in enumerate(ix):scores[ids]=p[i];gf[ids]=float(g[i])
                summary[dataset][key][variant]=metrics(rows,scores)
                probabilities[variant]=scores;weights[variant]=gf
            with (output/(dataset+'_'+key+'.jsonl')).open('x') as f:
                for i,row in enumerate(rows):
                    line={**row,**{k:float(v[i]) for k,v in pred.items()},
                          'probabilities':{k:float(v[i]) for k,v in probabilities.items()},
                          'gates':{k:float(v[i]) for k,v in weights.items()}}
                    f.write(json.dumps(line,ensure_ascii=False,allow_nan=False)+'\n')
        dump(output/(dataset+'_metrics.json'),summary[dataset])
        print('TARGET_METRICS',dataset,json.dumps(summary[dataset]),flush=True)
    dump(output/'ALL_METRICS.json',summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','evaluate']);p.add_argument('config');p.add_argument('run')
    p.add_argument('--seeds',nargs='+',type=int,default=[2027,3407,4099]);p.add_argument('--groups',nargs='+',default=['Full'])
    a=p.parse_args();c=json.loads(Path(a.config).read_text())
    freeze(c,a.run,a.seeds,a.groups) if a.command=='freeze' else evaluate(c,a.run)
