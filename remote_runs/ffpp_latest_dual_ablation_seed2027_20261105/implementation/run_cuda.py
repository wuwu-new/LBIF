"""Finite remote CUDA stages. No scheduler, retries, CLIP execution or old tasks."""
import argparse
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
    prefix=stage.upper().replace('-','_');source=stage.startswith('source');preflight=stage.endswith('preflight')
    assert not any((run/(prefix+s)).exists() for s in ('_STARTED.json','_FAILURE.json','_COMPLETE.json'))
    with (Path(p.REMOTE)/'.gpu_exclusive.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from gpu_identity_guard import OwnedCudaGuard
        guard=OwnedCudaGuard(on_bind=lambda x:dump(run/(prefix+'_CUDA_OWNER.json'),x));guard.check()
        resources(run,p.configuration()['train_uint8_bytes']+p.GIB if source else 2*p.GIB)
        assert resources(run)['disk_free']>p.GIB+p.configuration()['output_allowance_bytes']
        lock=verify(run);registration=lock['registration_sha256']
        if stage=='source':exited(run,'SOURCE_PREFLIGHT',registration)
        if not source:
            done=exited(run,'SOURCE',registration)
            assert sha(run/'SELECTED_ENDPOINT_LOCK.json')==done['selection_lock_sha256']
            proof=read(run/'SOURCE_LOCAL_VERIFICATION.json')
            assert proof['passed'] and proof['registration_sha256']==registration
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
        def optimizer(m):return torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.01,betas=(.9,.999),eps=1e-8)
        def forward(m,xx,bb,tag):
            kwargs={'return_patch':tag in ('retained_training','selected_localization','real_TRAIN_gradient','disposable_synthetic')} if isinstance(m,PatchRegionDelta) else {}
            out=m(xx,bb,enabled=True,**kwargs);meter[tag]+=len(bb)
            assert bool(torch.isfinite(out['logit']).all()) and float(out['delta'].abs().max())<=.5
            return out
        def batch(rows):return torch.from_numpy(np.stack([pixels(r) for r in rows])).cuda().float()/255
        def load(endpoint):
            assert sha(endpoint['path'])==endpoint['sha256']
            ck=torch.load(endpoint['path'],map_location='cuda',weights_only=True)
            assert [ck[k] for k in ('arm','seed','epoch','updates','registration_sha256')]==[endpoint['arm'],endpoint['seed'],endpoint['epoch'],719*endpoint['epoch'],registration]
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
                network_frame_forwards=dict(meter),resources=resources(run))
            dump(run/(prefix+'_COMPLETE.json'),extra);state(run,stage+'_complete')

        if source and preflight:
            for part,rows in (('TRAIN',train),('VAL',val)):
                assert len({p.identity(r) for r in rows})==len(rows)
                for i,r in enumerate(rows):
                    assert sha(r['image_path'])==r['sha256']
                    assert sha(r['prior_aligned_image_path'])==r['prior_aligned_sha256']
                    if r['mask_path']:assert r['mask_valid'] and sha(r['mask_path'])==r['mask_sha256']
                    if i%4096==0:guard.check();state(run,'input_SHA_audit',part=part,done=i,total=len(rows))
            anchor_errors={};selected=read(run/'REUSED_SELECTED_ENDPOINTS.json')
            for seed in p.SEEDS:
                endpoint=selected[str(seed)];path=endpoint['checkpoint_path']
                assert sha(path)==endpoint['checkpoint_sha256']
                saved=torch.load(path,map_location='cuda',weights_only=True)
                assert [saved[k] for k in ('arm','seed','epoch','updates','registration_sha256')]==['LegacyPoolDelta',seed,endpoint['epoch'],719*endpoint['epoch'],endpoint['checkpoint_registration_sha256']]
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                with torch.device('cuda'):m=BoundaryDelta('LegacyPoolDelta')
                m.load_state_dict(saved['model'],strict=True);m.eval()
                assert tensor_sha(m.state_dict())==endpoint['model_sha256']
                anchor_errors[str(seed)]=source_probe(m,seed,run/'reused_selected_source'/f'{seed}.csv','historical_full_replay')
                del m,saved
            current_full_errors={};full_selected=read(run/'REFERENCE_FULL_ENDPOINTS.json')
            for seed in p.SEEDS:
                endpoint=full_selected[str(seed)]['PatchClassificationContrast']
                assert sha(endpoint['path'])==endpoint['sha256']
                saved=torch.load(endpoint['path'],map_location='cuda',weights_only=True)
                assert saved['registration_sha256']==p.PARENT_REG and saved['seed']==seed
                assert saved['arm']=='PatchClassificationContrast' and saved['updates']==719*endpoint['epoch']
                m=make(seed,'PatchClassificationContrast');m.load_state_dict(saved['model'],strict=True);m.eval()
                assert tensor_sha(m.state_dict())==endpoint['model_sha256']
                current_full_errors[str(seed)]=source_probe(m,seed,run/'reference_full_source'/f'{seed}.csv','latest_full_replay')
                del m,saved
            initial={};shared_initial={};checks={};oldinitial=read(run/'REUSED_INITIALIZATION.json')['initialization_sha256']
            latest_initial=read(run/'REFERENCE_FULL_INITIALIZATION.json')
            real_index=next(i for i,r in enumerate(train) if r['label']==0 and r['mask_valid'])
            fake_indices=[i for i,r in enumerate(train) if r['label']==1 and r['mask_valid']][:64]
            fake_index=None
            for i in fake_indices:
                mask=torch.from_numpy(mask_pixels(train[i])[None,None]).cuda().float()/255
                target,eligible,_=patch_targets(mask,torch.ones(1,dtype=torch.bool,device='cuda'),torch.ones(1,device='cuda'))
                positives=int(target.sum())
                if 2<=positives<=254:fake_index=i;break
            assert fake_index is not None,'No eligible mixed fake mask among64 frozen TRAIN candidates'
            ids=[real_index,fake_index];realxx=batch([train[i] for i in ids])
            realmm=torch.from_numpy(np.stack([mask_pixels(train[i])[None] for i in ids])).cuda().float()/255
            yy=torch.tensor([0.,1.],device='cuda');valid=torch.ones(2,dtype=torch.bool,device='cuda')
            partial=torch.tensor([True,False],device='cuda');altered=realmm.clone();altered[1]=float('nan')
            torch.manual_seed(101);xx=torch.rand(2,3,256,256,device='cuda')
            synthetic_mask=torch.zeros(2,1,256,256,device='cuda');synthetic_mask[1,:,64:p.configuration()['selected_reload_forwards'],64:p.configuration()['selected_reload_forwards']]=1
            for seed in p.SEEDS:
                initial[str(seed)]={};shared_initial[str(seed)]={};checks[str(seed)]={}
                semtrain=table(run/'semantic_train'/f'{seed}.csv',11492)
                for i in ids:
                    assert (semtrain[i]['video_id'],int(semtrain[i]['source_frame_number']))==p.identity(train[i])
                    assert float(semtrain[i]['baseline'])==p.f32(float(semtrain[i]['semantic'])/2)
                bb=torch.tensor([float(semtrain[i]['baseline']) for i in ids],device='cuda',requires_grad=True)
                for arm in p.NEW_ARMS:
                    m=make(seed,arm);digest=tensor_sha(m.state_dict());initial[str(seed)][arm]=digest
                    shared=tensor_sha({k:v for k,v in m.state_dict().items() if not k.startswith('patch_classifier.')})
                    assert shared==oldinitial[str(seed)],'Historical shared constructor changed'
                    shared_initial[str(seed)][arm]=shared
                    assert digest==latest_initial[str(seed)]['PatchClassificationContrast'],'Full latest initialization changed'
                    out=forward(m,xx,bb,'synthetic_zero_head')
                    assert torch.equal(out['logit'],bb) and not bool(out['delta'].any())
                    assert float(m.error_map(torch.ones_like(xx)).abs().max())<=2e-7
                    out=forward(m,realxx,bb,'real_TRAIN_gradient')
                    terms=objective(m,out,yy,realmm,valid,split='TRAIN')
                    params=(m.encoder[0][0].weight,m.patch_classifier.weight,m.decoder[2].weight,bb)
                    pg=torch.autograd.grad(terms['patch_classification'],params,allow_unused=True,retain_graph=True)
                    assert all(g is not None and float(g.abs().sum())>0 for g in pg[:2]) and pg[2:]==(None,None)
                    cg=torch.autograd.grad(terms['patch_contrast'],params,allow_unused=True,retain_graph=True)
                    assert cg[0] is not None and float(cg[0].abs().sum())>0 and cg[1:]==(None,None,None)
                    bg=torch.autograd.grad(terms['boundary'],list(m.decoder.parameters()),retain_graph=True)
                    assert all(g is not None for g in bg) and any(float(g.abs().sum())>0 for g in bg)
                    assert torch.autograd.grad(terms['total'],bb,allow_unused=True,retain_graph=True)[0] is None
                    weights=p.variant(arm)
                    expected=(terms['image']+.2*terms['boundary']+.01*terms['regularizer']+
                        weights['patch_classification']*terms['patch_classification']+
                        weights['patch_contrast']*terms['patch_contrast'])
                    assert torch.equal(terms['total'],expected),'Changed loss mixture'
                    total_patch=torch.autograd.grad(terms['total'],m.patch_classifier.weight,retain_graph=True)[0]
                    assert bool(total_patch.abs().sum()>0)==bool(weights['patch_classification'])
                    if arm=='UniformPoolFull':
                        assert torch.equal(out['weight_mass'],torch.ones_like(out['weight_mass']))
                        detached=torch.autograd.grad(terms['image'],m.decoder[2].weight,allow_unused=True,retain_graph=True)[0]
                        assert detached is None,'Uniform readout accidentally depends on decoder'
                    a=objective(m,out,yy,realmm,partial,split='TRAIN')
                    b=objective(m,out,yy,altered,partial,split='TRAIN')
                    assert all(torch.equal(a[k],b[k]) for k in a)
                    for split in ('VAL','TEST','target'):
                        try:objective(m,out,yy,realmm,valid,split=split)
                        except ValueError:pass
                        else:raise AssertionError('NonTRAIN supervision accepted')
                    del m,out,terms,pg,cg,bg,a,b
                    m=make(seed,arm);opt=optimizer(m)
                    for _ in range(2):
                        opt.zero_grad(set_to_none=True);out=forward(m,xx,bb,'disposable_synthetic')
                        terms=objective(m,out,yy,synthetic_mask,valid,split='TRAIN')
                        assert bool(torch.isfinite(terms['total']));terms['total'].backward();assert bb.grad is None
                        assert all(t.grad is not None for t in m.parameters())
                        torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
                    assert len(opt.state)==len(list(m.parameters())) and all(int(v['step'])==2 for v in opt.state.values())
                    checks[str(seed)][arm]=dict(passed=True,real_TRAIN_patch_encoder_gradient=True,
                        real_TRAIN_patch_classifier_gradient=True,real_TRAIN_contrast_encoder_gradient=True,
                        semantic_gradient=False,invalid_mask_invariant=True,nonTRAIN_supervision_rejected=True,
                        declared_variant=p.variant(arm),total_patch_gradient_enabled=bool(weights['patch_classification']),
                        disposable_updates=2,allocated_parameters=sum(t.numel() for t in m.parameters()))
                    del m,opt,out,terms;gc.collect();torch.cuda.empty_cache();guard.check()
            assert len({v[p.NEW_ARMS[0]] for v in initial.values()})==len(p.SEEDS)
            assert all(len(set(v.values()))==1 for v in initial.values()),'Unpaired initialization'
            n=len(p.SEEDS)*len(p.NEW_ARMS)
            assert dict(meter)==dict(historical_full_replay=len(p.SEEDS)*64,latest_full_replay=len(p.SEEDS)*64,
                synthetic_zero_head=n*2,real_TRAIN_gradient=n*2,disposable_synthetic=n*4)
            dump(run/'INITIALIZATION.json',initial)
            finish(dict(passed=True,historical_full_max_errors=anchor_errors,latest_full_max_errors=current_full_errors,initialization_sha256=initial,
                shared_initialization_sha256=shared_initial,gradient_checks=checks,
                real_TRAIN_fixture_indices=ids,real_TRAIN_fixture_identities=[p.identity(train[i]) for i in ids],
                disposable_optimizer_updates=n*2,new_retained_updates=0,
                preflight_frame_forwards=p.configuration()['source_preflight_total_forwards'],new_semantic_execution=False));return

        if source:
            contract=read(run/'SOURCE_PREFLIGHT_COMPLETE.json');assert contract['disposable_optimizer_updates']==p.configuration()['preflight_disposable_updates']
            images=np.empty((11492,3,256,256),np.uint8);masks=np.zeros((11492,1,256,256),np.uint8)
            available=np.array([r['mask_valid'] for r in train],bool);labels=np.array([r['label'] for r in train],np.float32)
            for i,r in enumerate(train):
                images[i]=pixels(r)
                if r['mask_valid']:masks[i,0]=mask_pixels(r)
                if i%2048==0:guard.check();state(run,'TRAIN_uint8_loading',done=i,total=len(train))
            endpoints={}
            for seed in p.SEEDS:
                semtrain=table(run/'semantic_train'/f'{seed}.csv',11492)
                for r,s in zip(train,semtrain):
                    assert (s['video_id'],int(s['source_frame_number']))==p.identity(r)
                    assert float(s['baseline'])==p.f32(float(s['semantic'])/2)
                bases=np.array([float(r['baseline']) for r in semtrain],np.float32)
                endpoints[str(seed)]={}
                for arm in p.NEW_ARMS:
                    folder=run/'source'/str(seed)/arm;folder.mkdir(parents=True)
                    m=make(seed,arm);assert tensor_sha(m.state_dict())==contract['initialization_sha256'][str(seed)][arm]
                    opt=optimizer(m);dump(folder/'BASELINE.json',p.metrics(val,valbase[seed])[0]);records=[]
                    for epoch in range(1,11):
                        m.train();schedule=p.sampling(train,epoch,seed);totals=[0.]*5;coverage=[0.,0.,0.];seen=set()
                        for step,ids in enumerate(schedule,1):
                            xx=torch.from_numpy(images[ids]).cuda().float()/255;bb=torch.from_numpy(bases[ids]).cuda()
                            yy=torch.from_numpy(labels[ids]).cuda();mm=torch.from_numpy(masks[ids]).cuda().float()/255
                            valid=torch.from_numpy(available[ids]).cuda();opt.zero_grad(set_to_none=True)
                            out=forward(m,xx,bb,'retained_training');terms=objective(m,out,yy,mm,valid,split='TRAIN')
                            loss=terms['total'];bc=terms['image'];aux=terms['boundary'];penalty=terms['regularizer']
                            assert bool(torch.isfinite(loss));loss.backward()
                            assert all(t.grad is not None for t in m.parameters())
                            torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
                            for j,key in enumerate(('image','boundary','regularizer','patch_classification','patch_contrast')):totals[j]+=float(terms[key].detach())
                            for j,key in enumerate(('eligible_patches','eligible_fake_patches','mixed_patches')):coverage[j]+=float(terms[key])
                            seen.update(ids)
                            if step%100==0 or step==719:guard.check();state(run,'training',seed=seed,arm=arm,epoch=epoch,step=step,total=719,loss=float(loss.detach()))
                        dump(folder/f'EXPOSURE_{epoch:02}.json',dict(seed=seed,arm=arm,epoch=epoch,updates=719,trajectory_updates=719*epoch,
                            schedule_sha256=p.fingerprint(schedule),unique_frames=len(seen),loss_means=[x/719 for x in totals],
                            loss_names=['image','boundary','regularizer','patch_classification','patch_contrast'],
                            patch_coverage_totals=dict(zip(('eligible','fake','mixed'),coverage)),patch_weights=p.variant(arm)))
                        checkpoint=folder/f'epoch_{epoch:02}.pt'
                        with checkpoint.open('xb') as f:torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),arm=arm,seed=seed,epoch=epoch,
                            updates=719*epoch,registration_sha256=registration,initialization_sha256=contract['initialization_sha256'][str(seed)][arm]),f)
                        saved=torch.load(checkpoint,map_location='cuda',weights_only=True)
                        assert tensor_sha(saved['model'])==tensor_sha(m.state_dict())
                        assert len(saved['optimizer']['state'])==len(list(m.parameters())) and all(int(v['step'])==719*epoch for v in saved['optimizer']['state'].values())
                        endpoint=dict(path=str(checkpoint),sha256=sha(checkpoint),model_sha256=tensor_sha(m.state_dict()),seed=seed,arm=arm,epoch=epoch,updates=719*epoch,registration_sha256=registration)
                        del saved
                        m.eval();logits=[];evaluation=folder/f'evaluation_{epoch:02}';evaluation.mkdir();fp=evaluation/'FRAMES.csv'
                        with torch.no_grad(),fp.open('x',newline='') as f:
                            writer=csv.writer(f);writer.writerow(['index','baseline','delta','fused','boundary_mass','pooled_norm'])
                            for start in range(0,22080,16):
                                rows=val[start:start+16];xx=batch(rows);bb=torch.tensor(valbase[seed][start:start+len(rows)],device='cuda')
                                out=forward(m,xx,bb,'full_source_VAL');z,d,mass,norm=[out[k].cpu().tolist() for k in ('logit','delta','weight_mass','pooled_norm')];logits.extend(z)
                                assert all(p.f32(valbase[seed][start+i]+d[i])==z[i] for i in range(len(rows)))
                                writer.writerows((start+i,valbase[seed][start+i],d[i],z[i],mass[i],norm[i]) for i in range(len(rows)))
                                if start%4096==0:guard.check();state(run,'source_VAL',seed=seed,arm=arm,epoch=epoch,done=start+len(rows),total=22080)
                        metrics,videos=p.metrics(val,logits);dump(evaluation/'VIDEOS.json',videos)
                        rec=dict(**endpoint,metrics=metrics,frames_path=str(fp),frames_sha256=sha(fp),videos_sha256=sha(evaluation/'VIDEOS.json'))
                        dump(evaluation/'RESULT.json',rec);records.append(rec)
                    best=p.select(records);dump(folder/'ALL_EPOCHS.json',records);dump(folder/'SELECTED.json',best)
                    replay=load(best);err=source_probe(replay,seed,best['frames_path'],'selected_reload')
                    dump(folder/'SELECTED_REPLAY.json',dict(passed=True,max_logit_error=err,frames=64,epoch=best['epoch']))
                    if arm!='UniformNoAux':
                        summary=collections.defaultdict(lambda:dict(frames=0,GT_pixels=0,predicted=0,precision_hits=0,recall_hits=0,response_sum=0.,patch_TP=0,patch_FP=0,patch_FN=0,patch_TN=0,patch_mixed=0))
                        with torch.no_grad():
                            for start in range(0,22080,16):
                                rows=val[start:start+16];xx=batch(rows);bb=torch.tensor(valbase[seed][start:start+len(rows)],device='cuda')
                                out=forward(replay,xx,bb,'selected_localization')
                                for j,r in enumerate(rows):
                                    if not r['mask_valid']:continue
                                    mask=torch.from_numpy(mask_pixels(r)[None,None]).cuda().float()/255
                                    gt=boundary_target(mask,replay.kernel)>.1
                                    pt,_,cov=patch_targets(mask,torch.ones(1,dtype=torch.bool,device='cuda'),torch.tensor([r['label']],device='cuda'))
                                    pp=out['patch_logit'][j:j+1].sigmoid()>=.5;truth=pt>.5
                                    pred=out['boundary_logit'][j:j+1].sigmoid()>=.5
                                    dg=F.max_pool2d(gt.float(),5,1,2)>0;dp=F.max_pool2d(pred.float(),5,1,2)>0;s=summary[r['method']]
                                    s['frames']+=1;s['GT_pixels']+=int(gt.sum());s['predicted']+=int(pred.sum())
                                    s['precision_hits']+=int((pred&dg).sum());s['recall_hits']+=int((gt&dp).sum())
                                    s['response_sum']+=float(out['boundary_logit'][j].sigmoid().mean())
                                    s['patch_TP']+=int((pp&truth).sum());s['patch_FP']+=int((pp&~truth).sum())
                                    s['patch_FN']+=int((~pp&truth).sum());s['patch_TN']+=int((~pp&~truth).sum())
                                    s['patch_mixed']+=int(((cov>0)&(cov<1)).sum())
                                if start%4096==0:guard.check();state(run,'selected_localization',seed=seed,arm=arm,done=start+len(rows),total=22080)
                        dump(folder/'LOCALIZATION.json',dict(threshold=.5,truth_threshold=.1,tolerance_pixels=2,eligible_only=True,per_method=dict(summary)))
                    endpoints[str(seed)][arm]=best
                    dump(folder/'COMPLETE.json',dict(passed=True,updates=7190,selected_epoch=best['epoch'],epoch10_metrics=records[-1]['metrics']))
                    del m,opt,replay,out,xx,bb,yy,mm,valid,loss,bc,aux,penalty;gc.collect();torch.cuda.empty_cache()
            assert meter['retained_training']==p.configuration()['retained_training_frame_forwards'] and meter['full_source_VAL']==p.configuration()['source_VAL_frame_forwards']
            assert meter['selected_localization']==p.configuration()['selected_localization_forwards'] and meter['selected_reload']==p.configuration()['selected_reload_forwards']
            dump(run/'SELECTED_ENDPOINT_LOCK.json',endpoints)
            finish(dict(passed=True,new_retained_updates=p.configuration()['retained_updates'],disposable_optimizer_updates=0,new_semantic_execution=False,
                selection_lock_sha256=sha(run/'SELECTED_ENDPOINT_LOCK.json'),target_results_for_selection=False));return

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
                for (seed,arm),m in models.items():
                    bb=torch.tensor([p.f32(float(r['s'+str(seed)])/2) for r in sem[:16]],device='cuda')
                    a=forward(m,xx,bb,'target_repeat_probe')['logit'];b=forward(m,xx,bb,'target_repeat_probe')['logit']
                    err=float((a-b).abs().max());assert err<=2e-7;repeats[domain][f'{seed}/{arm}']=err
                guard.check()
        assert sum(meter.values())==p.configuration()['target_preflight_forwards_per_invocation']
        proof=dict(passed=True,source_errors=source_errors,target_repeat_errors=repeats,preflight_forwards=p.configuration()['target_preflight_forwards_per_invocation'],new_optimizer_updates=0)
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
                    for start in range(0,len(rows),16):
                        chunk=rows[start:start+16];xx=target_batch(chunk);values={}
                        for (seed,arm),m in models.items():
                            baseline=[p.f32(float(r['s'+str(seed)])/2) for r in sem[start:start+len(chunk)]]
                            bb=torch.tensor(baseline,device='cuda');out=forward(m,xx,bb,'full_target')
                            d,z,mass,norm=[out[k].cpu().tolist() for k in ('delta','logit','weight_mass','pooled_norm')]
                            assert all(p.f32(b+d[i])==z[i] for i,b in enumerate(baseline))
                            values[f'{seed}_{arm}']=dict(baseline=baseline,delta=d,fused=z,mass=mass,norm=norm)
                        for j in range(len(chunk)):writer.writerow([start+j,*[values[k][field][j] for field in ('baseline','delta','fused','mass','norm') for k in keys]])
                        if (start+len(chunk))%2048==0 or start+len(chunk)==len(rows):f.flush();guard.check();state(run,'target',domain=domain,done=start+len(chunk),total=len(rows))
                dump(evaluation/(domain+'_COMPLETE.json'),dict(frames=len(rows),real_videos=nr,fake_videos=nf,residual_frame_forwards=len(p.SEEDS)*len(p.NEW_ARMS)*len(rows),frames_sha256=sha(path)))
        assert meter['full_target']==p.configuration()['full_target_residual_forwards']
        assert all(tensor_sha(m.state_dict())==endpoints[str(seed)][arm]['model_sha256'] for (seed,arm),m in models.items())
        finish(dict(passed=True,full_residual_forwards=p.configuration()['full_target_residual_forwards'],new_optimizer_updates=0,new_semantic_execution=False,target_masks=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--stage',choices=('source-preflight','source','target-preflight','target'),required=True);args=ap.parse_args()
    try:execute(args.run.resolve(),args.stage)
    except Exception:
        prefix=args.stage.upper().replace('-','_');path=args.run/(prefix+'_FAILURE.json')
        if args.run.is_dir() and not path.exists():dump(path,dict(time=stamp(),traceback=traceback.format_exc()))
        raise
