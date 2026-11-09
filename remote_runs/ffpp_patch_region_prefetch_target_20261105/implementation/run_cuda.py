"""Finite remote CUDA stages. No scheduler, retries, CLIP execution or old tasks."""
import argparse
from contextlib import closing
from prefetch import ordered_batches,target_arrays
import collections
import csv
import fcntl
from fractions import Fraction
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import time
import traceback
import plan as p
from legacy_support import read,dump,sha,stamp,resources,state,tensor_sha

def table(path,count):
    with Path(path).open(newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==count and [int(r['index']) for r in rows]==list(range(count))
    return rows

def verify(run):
    assert run==Path(p.REMOTE)/p.RUN
    lock=read(run/'REGISTRATION.json')
    assert lock['configuration']==p.configuration()
    assert lock['registration_sha256']==p.fingerprint({k:v for k,v in lock.items() if k!='registration_sha256'})
    for name,digest in lock['files'].items():assert sha(run/name)==digest,('Bundle changed',name)
    for name,digest in read(run/'EXTERNAL_LOCK.json').items():assert sha(name)==digest,('Dependency changed',name)
    return lock

def exited(run,prefix,registration):
    assert not (run/(prefix+'_FAILURE.json')).exists()
    done=read(run/(prefix+'_COMPLETE.json'))
    assert done['passed'] and done['registration_sha256']==registration
    launch=read(run/(prefix+'_LAUNCH.json'));proc=Path('/proc')/str(launch['pid'])/'cmdline'
    assert not proc.exists() or not proc.read_bytes(),'Previous stage still alive'
    return done

def execute(run,stage):
    assert stage in ('target-preflight','target'),'Target-only sidecar'
    prefix=stage.upper().replace('-','_');source=stage.startswith('source');preflight=stage.endswith('preflight')
    assert not any((run/(prefix+s)).exists() for s in ('_STARTED.json','_FAILURE.json','_COMPLETE.json'))
    with (Path(p.REMOTE)/'.gpu_exclusive.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from gpu_identity_guard import OwnedCudaGuard
        guard=OwnedCudaGuard(on_bind=lambda x:dump(run/(prefix+'_CUDA_OWNER.json'),x));guard.check()
        resources(run,2*p.GIB)
        assert resources(run)['disk_free']>p.GIB+p.configuration()['output_allowance_bytes']
        lock=verify(run);registration=lock['registration_sha256']
        if stage=='source':exited(run,'SOURCE_PREFLIGHT',registration)
        if not source:
            adoption=read(run/'SOURCE_ADOPTION.json')
            assert adoption['source_run']==p.SOURCE_RUN and adoption['source_registration_sha256']==p.SOURCE_REG
            for name,digest in adoption['files'].items():assert sha(run/name)==digest
            done=exited(run,'SOURCE',p.SOURCE_REG)
            assert sha(run/'SELECTED_ENDPOINT_LOCK.json')==done['selection_lock_sha256']
            proof=read(run/'SOURCE_LOCAL_VERIFICATION.json')
            assert proof['passed'] and proof['registration_sha256']==p.SOURCE_REG
            assert proof['selection_lock_sha256']==done['selection_lock_sha256']
            if not preflight:exited(run,'TARGET_PREFLIGHT',registration)
        train,val=[read(run/'inputs'/(name+'.json')) for name in ('fit','source_val')]
        assert (len(train),len(val))==(11492,22080)
        assert not ({r['family_id'] for r in train}&{r['family_id'] for r in val})
        dump(run/(prefix+'_STARTED.json'),dict(time=stamp(),registration_sha256=registration))
        import numpy as np
        import torch
        from torch.nn import functional as F
        from model_cuda import PatchRegionDelta,objective,patch_targets
        from legacy_model_cuda import BoundaryDelta
        from prior_cuda_model import boundary_target
        from checked_pixels import pixels,mask_pixels
        torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        torch.cuda.init();context=torch.empty(1,device='cuda');torch.cuda.synchronize();guard.check();del context
        start_time=time.monotonic();meter=collections.Counter()
        semantic=table(run/'semantic_source/VAL.csv',22080)
        valbase={s:[p.f32(float(r['s'+str(s)])/2) for r in semantic] for s in p.SEEDS}
        def make(seed,arm):
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            with torch.device('cuda'):m=PatchRegionDelta(arm)
            assert all(t.is_cuda for t in m.parameters());return m
        def forward(m,xx,bb,tag):
            kwargs={'return_patch':tag in ('retained_training','selected_localization','real_TRAIN_gradient','disposable_synthetic')} if isinstance(m,PatchRegionDelta) else {}
            out=m(xx,bb,enabled=True,**kwargs);meter[tag]+=len(bb)
            assert bool(torch.isfinite(out['logit']).all()) and float(out['delta'].abs().max())<=.5
            return out
        def batch(rows):return torch.from_numpy(np.stack([pixels(r) for r in rows])).cuda().float()/255
        def load(endpoint):
            assert sha(endpoint['path'])==endpoint['sha256']
            ck=torch.load(endpoint['path'],map_location='cuda',weights_only=True)
            assert [ck[k] for k in ('arm','seed','epoch','updates','registration_sha256')]==[endpoint['arm'],endpoint['seed'],endpoint['epoch'],719*endpoint['epoch'],p.SOURCE_REG]
            m=make(endpoint['seed'],endpoint['arm']);m.load_state_dict(ck['model'],strict=True)
            assert tensor_sha(m.state_dict())==tensor_sha(ck['model'])==endpoint['model_sha256']
            assert len(ck['optimizer']['state'])==len(list(m.parameters()))
            assert all(int(v['step'])==719*endpoint['epoch'] for v in ck['optimizer']['state'].values())
            del ck;return m.eval()
        def source_probe(m,seed,framepath,tag):
            expected=table(framepath,22080);maximum=0.
            with torch.no_grad():
                for start in range(0,64,16):
                    ids=p.PROBES[start:start+16];xx=batch([val[i] for i in ids]);bb=torch.tensor([valbase[seed][i] for i in ids],device='cuda')
                    z=forward(m,xx,bb,tag)['logit'].cpu().tolist()
                    maximum=max(maximum,max(abs(a-float(expected[i]['fused'])) for a,i in zip(z,ids)))
            assert maximum<=2e-7;return maximum
        def finish(extra):
            extra.update(time=stamp(),registration_sha256=registration,seconds=time.monotonic()-start_time,
                network_frame_forwards=dict(meter),resources=resources(run),source_registration_sha256=p.SOURCE_REG,
                source_adoption_sha256=sha(run/'SOURCE_ADOPTION.json'))
            dump(run/(prefix+'_COMPLETE.json'),extra);state(run,stage+'_complete')

        from PIL import Image
        endpoints=read(run/'SELECTED_ENDPOINT_LOCK.json');models={};source_errors={}
        for seed in p.SEEDS:
            for arm in p.NEW_ARMS:
                endpoint=endpoints[str(seed)][arm];m=load(endpoint);models[(seed,arm)]=m
                source_errors[f'{seed}/{arm}']=source_probe(m,seed,endpoint['frames_path'],'source_reload_probe')
        pixel_hashes=read(read(run/'PIXEL_REFERENCE.json')['path'])
        def target_batch(rows):
            arrays=[]
            for r in rows:
                path=Path(r['image_path'])
                with path.open('rb') as f:
                    raw=f.read();assert hashlib.sha256(raw).hexdigest()==pixel_hashes[str(path)]
                    if hasattr(os,'posix_fadvise'):os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
                with Image.open(io.BytesIO(raw)) as image:
                    rgb=image.convert('RGB')
                    if rgb.size!=(256,256):rgb=rgb.resize((256,256),Image.Resampling.BICUBIC)
                    arrays.append(np.array(rgb,dtype=np.uint8).transpose(2,0,1).copy())
            return torch.from_numpy(np.stack(arrays)).cuda().float()/255
        repeats={}
        with torch.no_grad():
            for domain in p.DOMAINS:
                rows=read(run/'manifests'/(domain+'.json'));sem=table(run/'semantic_target'/(domain+'.csv'),len(rows));xx=target_batch(rows[:16]);repeats[domain]={}
                with closing(ordered_batches(rows[:16],lambda rr:target_arrays(rr,pixel_hashes))) as batches:
                    _,_,arrays=next(batches);parallel_xx=torch.from_numpy(arrays).cuda().float()/255
                assert torch.equal(xx,parallel_xx),'Changed target pixels'
                for (seed,arm),m in models.items():
                    bb=torch.tensor([p.f32(float(r['s'+str(seed)])/2) for r in sem[:16]],device='cuda')
                    a=forward(m,xx,bb,'target_repeat_probe');b=forward(m,parallel_xx,bb,'target_repeat_probe')
                    err=max(float((a[k]-b[k]).abs().max()) for k in ('logit','delta','weight_mass','pooled_norm'))
                    assert err<=2e-7;repeats[domain][f'{seed}/{arm}']=err
                guard.check()
        assert sum(meter.values())==864
        proof=dict(passed=True,source_errors=source_errors,target_repeat_errors=repeats,preflight_forwards=864,new_optimizer_updates=0)
        if preflight:
            benchmark_rows=val[:512];baseline_results=None;trials=[];maximum=0.
            for mode in (False,True,True,False):
                torch.cuda.synchronize();began=time.monotonic();results=[]
                with closing(ordered_batches(benchmark_rows,lambda rr:np.stack([pixels(r) for r in rr]),parallel=mode)) as batches:
                    with torch.no_grad():
                        for start,chunk,arrays in batches:
                            xx=torch.from_numpy(arrays).cuda().float()/255;batch_results=[]
                            for (seed,arm),m in models.items():
                                bb=torch.tensor(valbase[seed][start:start+len(chunk)],device='cuda')
                                out=forward(m,xx,bb,'acceleration_benchmark')
                                batch_results.extend(out[k].cpu().tolist() for k in ('delta','logit','weight_mass','pooled_norm'))
                            results.append(batch_results)
                torch.cuda.synchronize();elapsed=time.monotonic()-began
                if baseline_results is None:baseline_results=results
                else:
                    for aa,bb in zip(baseline_results,results):
                        assert len(aa)==len(bb)
                        for a,b in zip(aa,bb):
                            assert len(a)==len(b)
                            maximum=max(maximum,max(abs(x-y) for x,y in zip(a,b)))
                    assert maximum<=2e-7
                trials.append(dict(parallel=mode,seconds=elapsed,frames=512,model_forwards=1536))
                guard.check()
            ratio=sum(t['seconds'] for t in trials if not t['parallel'])/sum(t['seconds'] for t in trials if t['parallel'])
            choice=dict(passed=True,trials=trials,speedup=ratio,enable_prefetch=ratio>=1.05,
                max_output_error=maximum,extra_frame_forwards=6144,new_optimizer_updates=0,
                source_registration_sha256=p.SOURCE_REG,selection_lock_sha256=adoption['selection_lock_sha256'])
            dump(run/'ACCELERATION_DECISION.json',choice)
            assert meter['acceleration_benchmark']==6144
            proof.update(preflight_forwards=7008,acceleration=choice)
        else:
            choice=read(run/'ACCELERATION_DECISION.json')
            assert choice['passed'] and choice['max_output_error']<=2e-7
            assert choice['source_registration_sha256']==p.SOURCE_REG
            assert choice['selection_lock_sha256']==adoption['selection_lock_sha256']
            assert choice['enable_prefetch']==(choice['speedup']>=1.05)
        dump(run/(prefix+'_CONTRACT.json'),proof)
        if preflight:finish(proof);return
        evaluation=run/'target';evaluation.mkdir()
        with torch.no_grad():
            for domain,(nr,nf) in zip(p.DOMAINS,p.COUNTS):
                rows=read(run/'manifests'/(domain+'.json'));assert len(rows)==32*(nr+nf)
                sem=table(run/'semantic_target'/(domain+'.csv'),len(rows));path=evaluation/(domain+'_FRAMES.csv')
                keys=[f'{s}_{a}' for s in p.SEEDS for a in p.NEW_ARMS]
                with path.open('x',newline='') as f:
                    writer=csv.writer(f);writer.writerow(['index',*[k+'_'+field for field in ('baseline','delta','fused','mass','norm') for k in keys]])
                    with closing(ordered_batches(rows,lambda rr:target_arrays(rr,pixel_hashes),parallel=choice['enable_prefetch'])) as batches:
                        for start,chunk,arrays in batches:
                            xx=torch.from_numpy(arrays).cuda().float()/255;values={}
                            for (seed,arm),m in models.items():
                                baseline=[p.f32(float(r['s'+str(seed)])/2) for r in sem[start:start+len(chunk)]]
                                bb=torch.tensor(baseline,device='cuda');out=forward(m,xx,bb,'full_target')
                                d,z,mass,norm=[out[k].cpu().tolist() for k in ('delta','logit','weight_mass','pooled_norm')]
                                assert all(p.f32(b+d[i])==z[i] for i,b in enumerate(baseline))
                                values[f'{seed}_{arm}']=dict(baseline=baseline,delta=d,fused=z,mass=mass,norm=norm)
                            for j in range(len(chunk)):writer.writerow([start+j,*[values[k][field][j] for field in ('baseline','delta','fused','mass','norm') for k in keys]])
                            if (start+len(chunk))%2048==0 or start+len(chunk)==len(rows):f.flush();guard.check();state(run,'target',domain=domain,done=start+len(chunk),total=len(rows))
                dump(evaluation/(domain+'_COMPLETE.json'),dict(frames=len(rows),real_videos=nr,fake_videos=nf,residual_frame_forwards=3*len(rows),frames_sha256=sha(path)))
        assert meter['full_target']==669408
        assert all(tensor_sha(m.state_dict())==endpoints[str(seed)][arm]['model_sha256'] for (seed,arm),m in models.items())
        finish(dict(passed=True,full_residual_forwards=669408,new_optimizer_updates=0,new_semantic_execution=False,target_masks=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--stage',choices=('target-preflight','target'),required=True);args=ap.parse_args()
    try:execute(args.run.resolve(),args.stage)
    except Exception:
        prefix=args.stage.upper().replace('-','_');path=args.run/(prefix+'_FAILURE.json')
        if args.run.is_dir() and not path.exists():dump(path,dict(time=stamp(),traceback=traceback.format_exc()))
        raise
