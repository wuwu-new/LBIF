"""Two new correction trajectories; exact historical objective and ten epochs."""
import argparse
import csv
import fcntl
import gc
from pathlib import Path
import time
import traceback
import confirmation_plan as q
from common_cuda import read,sha,dump,stamp,resources,state,verify,tensor_hash,environment


def run_source(run,preflight):
    prefix='SOURCE_PREFLIGHT' if preflight else 'SOURCE_TRAIN'
    assert not any((run/n).exists() for n in (prefix+'_STARTED.json',prefix+'_FAILURE.json','SOURCE_COMPLETE.json'))
    with (Path(q.REMOTE)/'.gpu_exclusive.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from gpu_identity_guard import OwnedCudaGuard
        guard=OwnedCudaGuard(on_bind=lambda x:dump(run/(prefix+'_CUDA_OWNER.json'),x));guard.check()
        lock=verify(run);resources(run,2*1024**3,2*1024**3)
        if not preflight:
            assert read(run/'SOURCE_PREFLIGHT.json')['passed']
            proc=Path('/proc')/str(read(run/'SOURCE_PREFLIGHT_LAUNCH.json')['pid'])/'cmdline';assert not proc.exists() or not proc.read_bytes()
        dump(run/(prefix+'_STARTED.json'),dict(time=stamp(),registration_sha256=lock['registration_sha256']))
        torch,pipeline,joint,make,build_encoder,encode,load,spec=environment(run,guard)
        from torch.nn import functional as F
        from torch.utils.data import WeightedRandomSampler
        from crosslayer_cache import DiskLayers,train_cache,isolated_rng
        from disk_val_cache import DiskArray
        original_pixels=pipeline.Pixels
        class ReleasedPixels(original_pixels):
            def __getitem__(self,index):
                value=super().__getitem__(index)
                import os
                with Path(self.rows[index]['image_path']).open('rb') as f:os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
                return value
        pipeline.Pixels=ReleasedPixels
        train,val=(read(run/'inputs'/f'{name}.json') for name in ('fit','source_val'))
        assert len(train)==11492 and len(val)==22080
        oldroot=Path(q.REMOTE)/q.PARENT;oldval=Path(q.REMOTE)/'v22_frame_coverage_fix1_20261101/VAL_DISK'
        valcache=dict(tokens=DiskLayers(oldroot/'VAL_EARLY',oldval,len(val)),rgb=DiskArray(oldval/'rgb.bin',len(val),(3,224,224),'uint8',1))
        started=time.monotonic();config=read(run/'V5_CONFIG.json');encoder=build_encoder()
        cache_errors=[]
        with isolated_rng(),torch.no_grad():
            for start in (0,11040):
                rgb=torch.stack([pipeline.Pixels(val[start:start+16])[i] for i in range(16)])
                fresh=encode(encoder,rgb).cpu();prior=valcache['tokens'][start:start+16]
                error=float((fresh-prior).abs().max());assert error<=2e-7
                assert torch.equal((rgb*255).round().to(torch.uint8),valcache['rgb'][start:start+16]);cache_errors.append(error)
        old=read(run/'OLD2027.json')
        with isolated_rng():initial2027=make(2027)
        assert tensor_hash(initial2027.state_dict())==old['initialization_sha256'],'Historical constructor/RNG mismatch'
        del initial2027;gc.collect();torch.cuda.empty_cache()
        model,oldhash=load(2027,old['path'],q.PARENT_REG,old['model_tensor_sha256'])
        source_errors=[]
        with torch.no_grad():
            for probe in read(run/'SOURCE_PROBES.json'):
                start=probe['start'];xx=valcache['rgb'][start:start+16].cuda().float()/255;tt=valcache['tokens'][start:start+16].cuda().float()
                got=model(xx,tt)['logit'].cpu().tolist();err=max(abs(a-b) for a,b in zip(got,probe['logits']));assert err<=2e-7;source_errors.append(err)
        del model;gc.collect();torch.cuda.empty_cache()
        inits={};fixed_inputs={};fixture_steps={};fixture_loss={}
        xx=valcache['rgb'][:16].cuda().float()/255;tt=valcache['tokens'][:16].cuda().float();yy=torch.tensor([r['label'] for r in val[:16]],device='cuda',dtype=torch.float32)
        for seed in q.NEW_SEEDS:
            model=make(seed);inits[str(seed)]=tensor_hash(model.state_dict());fixed_inputs[str(seed)]=tensor_hash(model.token_input.state_dict())
            model.eval();xx.requires_grad_(True);result=model(xx,tt);swapped=model(torch.flip(xx,[-1]),tt)
            assert torch.equal(result['logit'],tt[:,0,0]) and torch.equal(result['logit'],swapped['logit'])
            assert torch.autograd.grad(result['residual_logit'].sum(),xx,allow_unused=True)[0] is None
            params=[x for x in model.parameters() if x.requires_grad]
            optimizer=torch.optim.AdamW([{'params':params,'lr':.0003},{'params':[],'lr':.00003}],weight_decay=.01)
            model.train();losses=[]
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True);out=model(xx,tt)
                loss=F.binary_cross_entropy_with_logits(out['logit'],yy)+.1*F.binary_cross_entropy_with_logits(out['residual_logit'],yy)
                loss.backward();assert all(t.grad is None for t in model.base.semantic.parameters())
                assert model.base.encoder.adapter.weight.grad.abs().sum()>0 and model.base.correction[-1].weight.grad.abs().sum()>0
                torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
            assert all(float(v['step'])==2 for v in optimizer.state.values());fixture_steps[str(seed)]=2;fixture_loss[str(seed)]=losses
            del model,optimizer,params,result,swapped,out,loss;gc.collect();torch.cuda.empty_cache();guard.check()
        assert len(set(inits.values()))==2 and len(set(fixed_inputs.values()))==1
        xx=tt=yy=None
        proof=dict(time=stamp(),passed=True,registration_sha256=lock['registration_sha256'],models_cuda_only=True,
            owned_nvml_pid=guard.owned_nvml_pid,old2027_model_tensor_sha256=oldhash,source_logit_max_error=max(source_errors),
            fresh_cache_max_error=max(cache_errors),source_predict_frame_forwards=192,source_encoding_probe_frames=32,
            initialization_sha256=inits,fixed_token_input_sha256=fixed_inputs,disposable_updates=4,fixture_steps=fixture_steps,fixture_loss=fixture_loss,
            source_TEST=False,target_access=False,resources=resources(run))
        dump(run/('SOURCE_PREFLIGHT.json' if preflight else 'SOURCE_RUNTIME_PREFLIGHT.json'),proof)
        if preflight:state(run,'source_preflight_passed');return
        resources(run,2*11492*(3*224*224+3*256*192*2+4)+2*1024**3)
        caches={'source_val':valcache}
        for name,aug in (('fit',False),('fit_aug',True)):
            state(run,'source_TRAIN_encoding',part=name,done=0,total=11492)
            caches[name]=train_cache(encoder,train,config,aug,guard,lambda _run:resources(run),run)
        del encoder;gc.collect();torch.cuda.empty_cache()
        def cache_identity():
            hashes={}
            for name in ('fit','fit_aug'):
                h=__import__('hashlib').sha256()
                for start in range(0,11492,128):
                    for value in (caches[name]['rgb'][start:start+128],caches[name]['tokens'].patches[start:start+128],caches[name]['tokens'].logits[start:start+128]):h.update(value.contiguous().numpy().tobytes())
                hashes[name]=h.hexdigest()
            return hashes
        cache_hashes=cache_identity();cache_checks={}
        dump(run/'TRAIN_CACHE_IDENTITY.json',dict(time=stamp(),sha256=cache_hashes,rows=11492,two_views_shared_readonly=True,new_disk_features=False))
        labels=torch.tensor([r['label'] for r in train],device='cuda',dtype=torch.float32)
        receipts={}
        def evaluate(model,folder,epoch):
            model.eval();path=folder/f'frames_{epoch:02}.csv'
            with path.open('x',newline='') as f:
                writer=csv.writer(f);writer.writerow(['index','logit'])
                with torch.no_grad():
                    for start in range(0,22080,16):
                        xx=valcache['rgb'][start:start+16].cuda().float()/255;tt=valcache['tokens'][start:start+16].cuda().float()
                        z=model(xx,tt)['logit'];assert bool(torch.isfinite(z).all())
                        for j,x in enumerate(z.cpu().tolist()):writer.writerow([start+j,x])
                        if start%2048==0:guard.check();resources(run)
            return path
        for seed in q.NEW_SEEDS:
            folder=run/'seeds'/str(seed);folder.mkdir(parents=True)
            model=make(seed);assert tensor_hash(model.state_dict())==inits[str(seed)]
            frozen_anchor_sha=tensor_hash(model.base.semantic.state_dict())
            assert sum(t.numel() for t in model.parameters() if t.requires_grad)==450818
            params=[t for t in model.parameters() if t.requires_grad]
            optimizer=torch.optim.AdamW([{'params':params,'lr':.0003},{'params':[],'lr':.00003}],weight_decay=.01)
            records=[]
            for epoch in range(11):
                e_start=time.monotonic();schedule=[];loss_sum=0.;steps=0;audits=[]
                if epoch:
                    model.train();schedule=list(WeightedRandomSampler(joint.sampler_weights(train),len(train),True,generator=torch.Generator().manual_seed(seed+10000*epoch)))
                    for offset in range(0,len(schedule),16):
                        ids=schedule[offset:offset+16];optimizer.zero_grad(set_to_none=True);outputs=[];losses=[]
                        for part in ('fit','fit_aug'):
                            result=model(caches[part]['rgb'][ids].cuda().float()/255,caches[part]['tokens'][ids].cuda().float());outputs.append(result)
                            losses.append(F.binary_cross_entropy_with_logits(result['logit'],labels[ids])+.1*F.binary_cross_entropy_with_logits(result['residual_logit'],labels[ids]))
                        loss=torch.stack(losses).mean()+.05*F.mse_loss(outputs[0]['delta'],outputs[1]['delta']);assert bool(torch.isfinite(loss));loss.backward()
                        if steps in (0,350,718):
                            grads={key:float(t.grad.norm()) if t.grad is not None else 0. for key,t in (('adapter',model.base.encoder.adapter.weight),('correction',model.base.correction[-1].weight),('readout',model.readout.project.weight))}
                            assert grads['adapter']>0 and grads['correction']>0 and (not(epoch>1 or steps>0) or grads['readout']>0)
                            assert all(t.grad is None for t in model.base.semantic.parameters());audits.append(dict(step=steps,gradients=grads))
                        torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step();steps+=1;loss_sum+=float(loss.detach())
                        if steps%100==0:guard.check();state(run,'source_training',seed=seed,epoch=epoch,updates=steps,total_updates=719)
                    assert steps==719 and all(float(v['step'])==719*epoch for v in optimizer.state.values())
                    outputs=losses=result=loss=None
                path=evaluate(model,folder,epoch);checkpoint=folder/f'epoch_{epoch:02}.pt'
                with checkpoint.open('xb') as f:torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),arm='CrossLayerNoImage',epoch=epoch,seed=seed,spec=spec,registration_sha256=lock['registration_sha256']),f)
                saved=torch.load(checkpoint,map_location='cuda',weights_only=True)
                identity=[saved[k] for k in ('arm','epoch','seed','registration_sha256')];model_sha=tensor_hash(model.state_dict())
                assert identity==['CrossLayerNoImage',epoch,seed,lock['registration_sha256']] and tensor_hash(saved['model'])==model_sha
                expected_steps=719*epoch;assert all(float(v['step'])==expected_steps for v in saved['optimizer']['state'].values())
                with isolated_rng():replay=make(seed)
                replay.load_state_dict(saved['model']);replay.eval()
                xx=valcache['rgb'][:16].cuda().float()/255;tt=valcache['tokens'][:16].cuda().float()
                with path.open(newline='') as f:expected=[float(row['logit']) for _,row in zip(range(16),csv.DictReader(f))]
                with torch.no_grad():error=max(abs(x-y) for x,y in zip(replay(xx,tt)['logit'].cpu().tolist(),expected))
                assert error<=2e-7
                record=dict(time=stamp(),seed=seed,epoch=epoch,updates=steps,total_updates=expected_steps,
                    initialization_sha256=inits[str(seed)],slot_sampling_sha256=q.fingerprint(schedule),schedule_size=len(schedule),
                    frame_scores=path.name,frames_sha256=sha(path),checkpoint=checkpoint.name,checkpoint_sha256=sha(checkpoint),
                    model_tensor_sha256=model_sha,identity=identity,checkpoint_logit_max_error=error,
                    gradient_audits=audits,loss_mean=loss_sum/max(1,steps),seconds=time.monotonic()-e_start)
                dump(folder/f'epoch_{epoch:02}.json',record);records.append(record)
                del saved,replay,xx,tt;gc.collect();torch.cuda.empty_cache();guard.check();state(run,'source_epoch_completed',seed=seed,epoch=epoch)
            assert tensor_hash(model.base.semantic.state_dict())==frozen_anchor_sha
            cache_checks[str(seed)]=cache_identity();assert cache_checks[str(seed)]==cache_hashes,'Shared TRAIN cache mutated'
            dump(folder/'READONLY_CACHE_CHECK.json',dict(passed=True,sha256=cache_checks[str(seed)],frozen_anchor_sha256=frozen_anchor_sha))
            receipts[str(seed)]=records[-1];params.clear();model=optimizer=None;gc.collect();torch.cuda.empty_cache()
        caches=labels=None;gc.collect();torch.cuda.empty_cache()
        report=dict(time=stamp(),registration_sha256=lock['registration_sha256'],seed_receipts=receipts,new_retained_updates=14380,
            source_training_seeds=list(q.NEW_SEEDS),source_TEST=False,target_access=False,disposable_updates_this_process=4,
            readonly_cache_checks=cache_checks,seconds=time.monotonic()-started,resources=resources(run))
        dump(run/'SOURCE_REPORT.json',report);dump(run/'SOURCE_COMPLETE.json',dict(time=stamp(),registration_sha256=lock['registration_sha256'],report_sha256=sha(run/'SOURCE_REPORT.json')))
        state(run,'source_complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    try:run_source(args.run.resolve(),args.preflight_only)
    except Exception:
        path=args.run/(('SOURCE_PREFLIGHT' if args.preflight_only else 'SOURCE_TRAIN')+'_FAILURE.json')
        if not path.exists():dump(path,dict(time=stamp(),traceback=traceback.format_exc()))
        raise
