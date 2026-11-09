"""Independent V18 source-only pilot; CUDA models, bounded cache lifecycle."""
import argparse
from array import array
from collections import defaultdict
import datetime
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time

import multiscale_plan as plan
from cache_lifecycle import release_read_cache,validate_frame
from gpu_identity_guard import OwnedCudaGuard

ROOT=Path(plan.REMOTE_ROOT)
GIB=1024**3
SLOT=3*224*224+256*192*2+4


def stamp():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def dump(path,value):
    with Path(path).open('x',encoding='utf-8') as handle:json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)


def state(run,stage,**values):
    record=dict(time=stamp(),stage=stage,**values)
    with (run/'.STATE.pending').open('w',encoding='utf-8') as handle:json.dump(record,handle,allow_nan=False)
    os.replace(run/'.STATE.pending',run/'STATE.json')
    with (run/'STAGES.jsonl').open('a',encoding='utf-8') as handle:handle.write(json.dumps(record,allow_nan=False)+'\n')
    print(json.dumps(record),flush=True)


def memory():
    base=Path('/sys/fs/cgroup/memory')
    values={key:int(value) for key,value in (line.split() for line in (base/'memory.stat').read_text().splitlines())}
    return dict(cap=int((base/'memory.limit_in_bytes').read_text()),used=int((base/'memory.usage_in_bytes').read_text()),
                cache=values['cache'],rss=values['rss'],inactive_file=values['inactive_file'],active_file=values['active_file'])


def resources(run,additional_memory=0,additional_disk=0):
    current=memory();free=shutil.disk_usage(run).free
    plan.require(current['used']+additional_memory+6*GIB<=current['cap'],'6GiB cgroup reserve')
    plan.require(free>=GIB+additional_disk,'1GiB disk reserve; no history cleanup')
    return dict(**current,disk_free=free,additional_memory=additional_memory)


def verify(run):
    lock=plan.read(run/'REGISTRATION.json')
    plan.require(lock['registration_sha256']==plan.fingerprint({k:v for k,v in lock.items() if k!='registration_sha256'}),'Registration changed')
    plan.require(lock['configuration']==plan.configuration(),'Protocol changed')
    for name,sha in lock['bundle_sha256'].items():plan.require(plan.digest(run/name)==sha,'Bundle changed: '+name)
    for name,sha in plan.read(run/'EXTERNAL_LOCK.json').items():plan.require(plan.digest(name)==sha,'Dependency changed: '+name)
    train,val=(plan.read(run/'inputs'/f'{name}.json') for name in ('fit','source_val'))
    plan.require(plan.verify_split(train,val)==plan.read(run/'SPLIT_PROOF.json'),'Split changed')
    return lock,train,val


def prepare(run,lock,train,val):
    before=memory();needed=(len(val)+2*len(train))*SLOT+3*GIB+6*GIB+GIB//2
    plan.require(hasattr(os,'posix_fadvise'),'Scoped cache advice unavailable')
    seen=set();paths=[];size=0;current=before
    for source in plan.read(run/'CACHE_SCOPE.json')['manifests']:
        plan.require(plan.digest(source['path'])==source['sha256'],'Consumed manifest changed')
        for row in plan.read(source['path']):
            if current['cap']-current['used']>=needed:break
            path=row['image_path']
            if path in seen:continue
            seen.add(path);size+=release_read_cache(path,{path});paths.append(path)
            if len(paths)%256==0:current=memory()
            if len(paths)%8192==0:state(run,'scoped_read_cache_preparation',files=len(paths),memory=current)
        if current['cap']-current['used']>=needed:break
    after=memory()
    record=dict(time=stamp(),registration_sha256=lock['registration_sha256'],before=before,after=after,
        required_free_bytes=needed,files_advised=len(paths),bytes_advised=size,paths_fingerprint=plan.fingerprint(paths),
        files_deleted=0,files_opened_read_only=True,global_cache_reclaim=False,models_loaded=False)
    dump(run/'CACHE_PREPARE.json',record)
    plan.require(after['cap']-after['used']>=needed,'Insufficient scoped reclaim; no global fallback')
    state(run,'cache_prepared',preparation=record)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only',action='store_true');mode.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args();run=args.run.resolve()
    plan.require(run==ROOT/plan.RUN_NAME,'Wrong run root')
    prefix='PREPARE' if args.prepare_only else 'PREFLIGHT' if args.preflight_only else 'TRAIN'
    plan.require(not (run/(prefix+'_STARTED.json')).exists() and not (run/'COMPLETE.json').exists(),'Existing stage preserved')
    with (ROOT/'.gpu_exclusive.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        guard=OwnedCudaGuard(on_bind=lambda record:print('CUDA_OWNER',json.dumps(record),flush=True));guard.check()
        lock,train,val=verify(run)
        dump(run/(prefix+'_STARTED.json'),dict(time=stamp(),registration_sha256=lock['registration_sha256']))
        if args.prepare_only:prepare(run,lock,train,val);return
        plan.require(plan.read(run/'CACHE_PREPARE.json')['registration_sha256']==lock['registration_sha256'],'Missing cache preparation')
        if not args.preflight_only:
            check=plan.read(run/'PREFLIGHT.json')
            plan.require(check['passed'] and check['registration_sha256']==lock['registration_sha256'],'Missing CUDA preflight')
        resources(run,2*GIB,768*1024**2)
        import torch
        from torch import nn
        from torch.nn import functional as F
        from torch.utils.data import DataLoader,Dataset,WeightedRandomSampler
        from PIL import Image,ImageFilter
        import io
        sys.path.insert(0,str(run/'implementation/v5'));sys.path.append(str(ROOT))
        import gpu_pipeline as p
        import full_source_v2 as compact
        import joint_search_v4b as joint
        import strong_anchor_v5 as historical
        from multiscale_cuda import upgrade,canonical_base_state,route_probe
        plan.require(torch.cuda.is_available(),'CUDA mandatory; no CPU fallback')
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        config=plan.read(run/'V5_CONFIG.json');cfg=plan.configuration()
        plan.require(config['batch_size']==cfg['batch_size']==16 and config['epochs']==cfg['epochs']==10,'Historical budget changed')
        spec={**config['model'],**joint.ARCHITECTURES['IR_LayerScale']}
        original_pixels=p.Pixels

        class ReleasedPixels(original_pixels):
            def __init__(self,rows,augmented=False):super().__init__(rows,augmented);self.allowed={row['image_path'] for row in rows}
            def __getitem__(self,index):
                result=super().__getitem__(index)
                release_read_cache(self.rows[index]['image_path'],self.allowed)
                return result
        p.Pixels=ReleasedPixels

        class DegradedPixels(Dataset):
            def __init__(self,rows):self.rows=rows;self.original=ReleasedPixels(rows)
            def __len__(self):return len(self.rows)
            def __getitem__(self,index):
                rgb=self.original[index]
                image=Image.fromarray((rgb.permute(1,2,0).numpy()*255).round().astype('uint8'))
                image=image.resize((192,192),Image.Resampling.BICUBIC).resize((224,224),Image.Resampling.BICUBIC)
                image=image.filter(ImageFilter.GaussianBlur(.35))
                with io.BytesIO() as buffer:
                    image.save(buffer,format='JPEG',quality=85);buffer.seek(0)
                    with Image.open(buffer) as decoded:
                        import numpy as np
                        values=np.asarray(decoded.convert('RGB')).copy()
                return torch.from_numpy(values).permute(2,0,1).float()/255

        anchor=compact.CompactSemantic(config['semantic_checkpoint']).cuda().eval()
        def make(arm):
            p.seed_all(plan.SEED)
            return upgrade(joint.JointDual(joint.SemanticExit(anchor),spec).cuda(),arm)
        def tensor_hash(items):
            import hashlib
            value=hashlib.sha256()
            for name,tensor in sorted(items.items()):
                value.update(name.encode());value.update(tensor.detach().contiguous().cpu().numpy().tobytes())
            return value.hexdigest()

        encoder=compact.CompactEncoder(config)
        cpu_rgb=torch.stack([ReleasedPixels(val[:16])[index] for index in range(16)])
        with torch.no_grad():tokens=compact.compact_encode(encoder,cpu_rgb)
        pixels=(cpu_rgb*255).round().to(torch.uint8).cuda().float()/255
        p.seed_all(plan.SEED);plain=joint.JointDual(joint.SemanticExit(anchor),spec).cuda().eval()
        original_hash=tensor_hash(plain.state_dict())
        with torch.no_grad():expected=plain(pixels,tokens)
        initialization=[];audits={}
        for arm in plan.ARMS:
            model=make(arm).eval()
            plan.require(tensor_hash(canonical_base_state(model))==original_hash,'Changed V5 initialization')
            initialization.append(tensor_hash(model.state_dict()))
            model.encoder.audit_requested=True
            result=model(pixels,tokens)
            for key in ('logit','semantic_logit','delta','residual_logit'):torch.testing.assert_close(result[key],expected[key],rtol=0,atol=0)
            geometry=model.encoder.last_diagnostics
            plan.require([item[-1] for item in geometry['tap_shapes']]==cfg['tap_grids'],'Wrong tap geometry')
            loss=F.binary_cross_entropy_with_logits(result['logit'],torch.tensor([row['label'] for row in val[:16]],device='cuda',dtype=torch.float32))+.1*F.binary_cross_entropy_with_logits(result['residual_logit'],torch.tensor([row['label'] for row in val[:16]],device='cuda',dtype=torch.float32))
            gradients=torch.autograd.grad(loss,[model.encoder.adapter.weight,model.correction[-1].weight],retain_graph=False)
            plan.require(all(bool(value.abs().sum()>0) for value in gradients),'Dead adapter/injection first-step gradient')
            route=route_probe(model.encoder)
            count=sum(value.numel() for value in model.encoder.adapter.parameters())
            plan.require(count==cfg['adapter_added_parameters'],'Adapter parameter budget changed')
            audits[arm]=dict(base_output_exact=True,tap_diagnostics=geometry,route_probe=route,added_parameters=count,
                trainable_parameters=sum(value.numel() for value in model.parameters() if value.requires_grad),
                first_step_gradient_l1=[float(value.abs().sum()) for value in gradients],semantic_frozen=all(not value.requires_grad for value in model.semantic.parameters()))
            del model,result,loss,gradients
        plan.require(len(set(initialization))==1,'Two arms do not share initialization')
        old_models={'OldV2':historical.reference_model(config),'ReusedV5IR':joint.load_model(ROOT/'strong_anchor_v5_20261027/screening_seed2027/IR_LayerScale',anchor).cuda().eval()}
        prior_probe=plan.read(run/'NUMERIC_REFERENCES.json')
        errors={}
        with torch.no_grad():
            for view,dataset in (('clean',ReleasedPixels(val[:16])),('degraded',DegradedPixels(val[:16]))):
                rgb=torch.stack([dataset[index] for index in range(16)])
                features=compact.compact_encode(encoder,rgb)
                x=(rgb*255).round().to(torch.uint8).cuda().float()/255
                for name,model in old_models.items():
                    actual=model(x,features)['logit'].sigmoid().cpu().tolist()
                    saved=prior_probe[view][name]
                    error=max(abs(value-saved[index]) for index,value in enumerate(actual))
                    plan.require(error<2e-7,'Old-source numeric preflight failed')
                    errors[view+'_'+name]=error
        guard.check()
        preflight=dict(time=stamp(),passed=True,registration_sha256=lock['registration_sha256'],audits=audits,
            original_rng_preserved=True,matched_initialization_sha256=initialization[0],numeric_errors=errors,
            owned_nvml_pid=guard.owned_nvml_pid,source_test_access=False,target_inference=False,
            resources=resources(run),models_cuda_only=True)
        del plain,expected,pixels,tokens,cpu_rgb,rgb,features,x,model,old_models
        gc.collect();torch.cuda.empty_cache()
        if args.preflight_only:dump(run/'PREFLIGHT.json',preflight);print(json.dumps(preflight),flush=True);return
        dump(run/'RUNTIME_PREFLIGHT.json',preflight)
        resources(run,(len(val)+2*len(train))*SLOT+2*GIB,768*1024**2)
        cache_part='source_val';original_log=p.log
        def cache_log(kind,**values):
            original_log(kind,**values)
            if kind=='COMPACT_RAM_CACHE':guard.check();state(run,'source_cache',part=cache_part,resources=resources(run),**values)
        p.log=cache_log
        state(run,'source_cache',part=cache_part,done=0,total=len(val))
        caches={'source_val':compact.compact_cache(encoder,val,config)}
        groups=defaultdict(list)
        for index,row in enumerate(val):groups[row['video_id']].append(index)
        indices=[groups[key] for key in sorted(groups)]
        plan.require(all(len(items)==32 for items in indices),'Invalid source denominator')
        evaluation=run/'evaluation';evaluation.mkdir()

        def video_records(predictions):
            output={}
            with torch.no_grad():
                for name,values in predictions.items():
                    s=torch.tensor([[values['s'][index] for index in items] for items in indices],device='cuda',dtype=torch.float32)
                    d=torch.tensor([[values['d'][index] for index in items] for items in indices],device='cuda',dtype=torch.float32)
                    scores=(s+d).sigmoid().mean(-1).cpu().tolist();semantic=s.sigmoid().mean(-1).cpu().tolist()
                    output[name]=[dict(video_id=val[items[0]]['video_id'],family_id=val[items[0]]['family_id'],label=val[items[0]]['label'],frames=32,
                        score=scores[j],semantic=semantic[j]) for j,items in enumerate(indices)]
            return output

        @torch.no_grad()
        def predict_cached(model,frame_path=None):
            model.eval();values={key:array('f') for key in ('s','d','r')}
            handle=frame_path.open('x',encoding='utf-8') if frame_path is not None else None
            try:
                for start in range(0,len(val),16):
                    result=model(caches['source_val']['rgb'][start:start+16].cuda().float()/255,caches['source_val']['tokens'][start:start+16].cuda().float())
                    batch={key:result[field].cpu().tolist() for key,field in [('s','semantic_logit'),('d','delta'),('r','residual_logit')]}
                    plan.require(all(bool(torch.isfinite(result[field]).all()) for field in ('semantic_logit','delta','residual_logit')),'Nonfinite source output')
                    for key,column in batch.items():values[key].extend(column)
                    if handle is not None:
                        probabilities=result['logit'].sigmoid().cpu().tolist()
                        for j,row in enumerate(val[start:start+16]):
                            handle.write(json.dumps({key:row[key] for key in ('video_id','family_id','label','source_frame_number')} |
                                {key:column[j] for key,column in batch.items()} | dict(probability=probabilities[j]),allow_nan=False)+'\n')
            finally:
                if handle is not None:handle.close()
            return video_records({'model':values})['model']

        baselines={};expected_by_id={row['video_id']:row for row in plan.read(run/'BASELINE_SOURCE_VIDEOS.json')}
        checks={}
        for name in ('OldV2','ReusedV5IR'):
            model=(historical.reference_model(config) if name=='OldV2' else joint.load_model(ROOT/'strong_anchor_v5_20261027/screening_seed2027/IR_LayerScale',anchor).cuda().eval())
            videos=predict_cached(model,evaluation/(name+'_clean_FRAMES.jsonl'));baselines[name]=videos
            dump(evaluation/(name+'_clean_VIDEOS.json'),videos)
            prior_name='OldV2_Fixed' if name=='OldV2' else 'screening_seed2027_IR_LayerScale_Fixed'
            error=max(abs(row['score']-expected_by_id[row['video_id']]['scores'][prior_name]) for row in videos)
            result_auc=plan.auc([row['label'] for row in videos],[row['score'] for row in videos])
            expected_auc=.9749396135265701 if name=='OldV2' else .9750052509976896
            plan.require(abs(result_auc-expected_auc)<1e-10 and error<2e-7,'Full baseline reproduction failed; training forbidden')
            checks[name]=dict(auc=result_auc,max_video_error=error)
            del model
        semantic_auc=plan.auc([row['label'] for row in baselines['OldV2']],[row['semantic'] for row in baselines['OldV2']])
        plan.require(abs(semantic_auc-.9741782188615837)<1e-10,'Full semantic reproduction failed; training forbidden')
        dump(run/'BASELINE_REPRODUCTION.json',dict(passed=True,semantic_auc=semantic_auc,checks=checks,videos=690,frames=22080))
        state(run,'full_baselines_reproduced',semantic_auc=semantic_auc,checks=checks)
        for part,augmented in (('fit',False),('fit_aug',True)):
            cache_part=part;state(run,'source_cache',part=part,done=0,total=len(train))
            caches[part]=compact.compact_cache(encoder,train,config,augmented)
        del encoder;gc.collect();torch.cuda.empty_cache()
        labels=torch.tensor([row['label'] for row in train],device='cuda',dtype=torch.float32)
        selections={};initializations=[];schedules=[]
        for arm in plan.ARMS:
            folder=run/'arms'/arm;folder.mkdir(parents=True)
            model=make(arm);initial=tensor_hash(model.state_dict());initializations.append(initial)
            params=[value for value in model.parameters() if value.requires_grad]
            dump(folder/'INITIALIZATION.json',dict(sha256=initial,parameters=sum(value.numel() for value in params)))
            filters=[model.front.delta] if model.front.delta.requires_grad else [];filter_ids={id(value) for value in filters}
            optimizer=torch.optim.AdamW([{'params':[value for value in params if id(value) not in filter_ids],'lr':config['dual_lr']},
                {'params':filters,'lr':config['dual_lr']*config['filter_lr_multiplier']}],weight_decay=config['weight_decay'])
            best=None;arm_schedules=[]
            for epoch in range(11):
                state(run,'training' if epoch else 'initial_evaluation',arm=arm,epoch=epoch,epochs=10)
                started=time.monotonic();updates=0;total=0.;audits=[]
                if epoch:
                    model.train();generator=torch.Generator().manual_seed(plan.SEED+10000*epoch)
                    sampled=list(WeightedRandomSampler(joint.sampler_weights(train),len(train),True,generator=generator));arm_schedules.append(plan.fingerprint(sampled))
                    for start in range(0,len(sampled),16):
                        ids=sampled[start:start+16];optimizer.zero_grad(set_to_none=True);outputs=[];losses=[];diagnostics=[]
                        audit_step=updates in (0,350,718)
                        for part in ('fit','fit_aug'):
                            model.encoder.audit_requested=audit_step
                            result=model(caches[part]['rgb'][ids].cuda().float()/255,caches[part]['tokens'][ids].cuda().float())
                            losses.append(F.binary_cross_entropy_with_logits(result['logit'],labels[ids])+config['residual_aux_weight']*F.binary_cross_entropy_with_logits(result['residual_logit'],labels[ids]))
                            outputs.append(result)
                            if audit_step:diagnostics.append(model.encoder.last_diagnostics)
                        loss=torch.stack(losses).mean()+config['consistency_weight']*F.mse_loss(outputs[0]['delta'],outputs[1]['delta'])
                        plan.require(bool(torch.isfinite(loss)),'Nonfinite V5 loss');loss.backward()
                        if audit_step:
                            gradients={name:float(sum((value.grad.square().sum() for key,value in model.named_parameters() if key.startswith(prefix) and value.grad is not None),torch.zeros((),device='cuda')).sqrt())
                                for name,prefix in [('encoder','encoder.body.'),('adapter','encoder.adapter.'),('injection','correction.3.')]}
                            plan.require(all(value>0 for value in gradients.values()),'Dead actual encoder/adapter/injection gradient')
                            plan.require(all(not value.requires_grad for value in model.semantic.parameters()),'Semantic was unfrozen')
                            audits.append(dict(step=updates,gradients=gradients,views=diagnostics,semantic_frozen=True))
                        nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
                        updates+=1;total+=float(loss.detach())
                        if updates%100==0:guard.check();state(run,'training',arm=arm,epoch=epoch,updates=updates,total_updates=719,loss_mean=total/updates,resources=resources(run))
                videos=predict_cached(model);dump(folder/f'epoch_{epoch:02}_VIDEOS.json',videos)
                result_auc=plan.auc([row['label'] for row in videos],[row['score'] for row in videos])
                if not epoch:plan.require(abs(result_auc-semantic_auc)<1e-10,'Zero adapter/head epoch0 differs from semantic')
                resources(run,additional_disk=24*1024**2)
                checkpoint=f'epoch_{epoch:02}.pt'
                with (folder/checkpoint).open('xb') as handle:torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),arm=arm,seed=plan.SEED,epoch=epoch,spec=spec,registration_sha256=lock['registration_sha256']),handle)
                record=dict(arm=arm,seed=plan.SEED,epoch=epoch,updates=updates,video_auc=result_auc,checkpoint=checkpoint,
                    checkpoint_sha256=plan.digest(folder/checkpoint),video_scores=f'epoch_{epoch:02}_VIDEOS.json',
                    video_scores_sha256=plan.digest(folder/f'epoch_{epoch:02}_VIDEOS.json'),schedule_sha256=arm_schedules[-1] if epoch else None,
                    gradient_audits=audits,loss_mean=total/max(updates,1),seconds=time.monotonic()-started)
                dump(folder/f'epoch_{epoch:02}.json',record)
                if best is None or result_auc>best['video_auc']:best=record
                guard.check();state(run,'epoch_completed',arm=arm,epoch=epoch,auc=result_auc,updates=updates,resources=resources(run))
            dump(folder/'SELECTION.json',best);selections[arm]=best;schedules.append(arm_schedules)
            del model,optimizer,params,filters,outputs,losses,result,loss
            gc.collect();torch.cuda.empty_cache()
        plan.require(len(set(initializations))==1 and schedules[0]==schedules[1],'Matched initialization/sampling failed')
        dump(run/'SELECTION_LOCK.json',dict(rule=cfg['endpoint_rule'],selections=selections,same_initialization=True,same_sampling=True,
            target_inference=False,source_test_access=False,promotion=False))
        columns=dict(baselines)
        columns['Semantic']=[dict(row,score=row['semantic']) for row in baselines['OldV2']]
        selected={};receipts={}
        for arm in plan.ARMS:
            for name,record in ((arm,selections[arm]),(arm+'_epoch10',plan.read(run/'arms'/arm/'epoch_10.json'))):
                path=run/'arms'/arm/record['video_scores'];columns[name]=plan.read(path)
                selected[name]=dict(path=str(run/'arms'/arm/record['checkpoint']),sha256=record['checkpoint_sha256'],arm=arm,epoch=record['epoch'])
                receipts[name]=dict(path=str(path),sha256=plan.digest(path),inference_repeated=False)
        clean=plan.merge_videos(columns);dump(evaluation/'clean_VIDEOS.json',clean);dump(run/'CLEAN_REUSE_RECEIPTS.json',receipts)
        dump(run/'EVAL_LOCK.json',dict(time=stamp(),selected=selected,constant_g=1,source_selection_sha256=plan.digest(run/'SELECTION_LOCK.json'),
            registration_sha256=lock['registration_sha256'],source_test_access=False,target_inference=False))
        # Drop every full RAM cache before streaming the second VAL view.
        del caches,labels,columns,baselines;gc.collect();torch.cuda.empty_cache()
        state(run,'source_statistics',view='clean')
        clean_summary=plan.statistics(clean,'reused source VAL, selected single residual seed; exploratory')
        dump(evaluation/'clean_SUMMARY.json',clean_summary)
        active={}
        for name,item in selected.items():
            plan.require(plan.digest(item['path'])==item['sha256'],'Selected weight changed')
            model=make(item['arm']);saved=torch.load(item['path'],map_location='cuda',weights_only=True)
            plan.require(saved['epoch']==item['epoch'] and saved['registration_sha256']==lock['registration_sha256'],'Selected checkpoint identity changed')
            model.load_state_dict(saved['model'],strict=True);active[name]=model.eval().requires_grad_(False);del saved
        encoder=compact.CompactEncoder(config)
        reuse=plan.read(run/'DEGRADED_REUSE.json');prior=Path(reuse['frames'])
        plan.require(plan.digest(prior)==reuse['frames_sha256'],'Saved degraded reference changed')
        predictions={name:{key:array('f') for key in ('s','d','r')} for name in ('OldV2','ReusedV5IR',*active)}
        offset=0;state(run,'degraded_streaming',done=0,total=len(val),full_dataset_cache=False)
        with prior.open(encoding='utf-8') as old,(evaluation/'degraded_FRAMES.jsonl').open('x',encoding='utf-8') as handle:
            for rgb in DataLoader(DegradedPixels(val),batch_size=16,num_workers=2,pin_memory=True):
                batch={}
                for j,row in enumerate(val[offset:offset+len(rgb)]):
                    saved=json.loads(next(old));validate_frame(saved,row,offset+j,models=reuse['saved_models'])
                    for name in ('OldV2','ReusedV5IR'):
                        for key,value in saved['scores'][name].items():batch.setdefault(name,{}).setdefault(key,[]).append(value)
                with torch.no_grad():
                    tokens=compact.compact_encode(encoder,rgb);x=(rgb*255).round().to(torch.uint8).cuda().float()/255
                    old_s=torch.tensor(batch['OldV2']['s'],device='cuda')
                    plan.require(float((tokens[:,0,0].sigmoid()-old_s.sigmoid()).abs().max())<2e-7,'Shared semantic/degradation changed')
                    for name,model in active.items():
                        result=model(x,tokens)
                        plan.require(all(bool(torch.isfinite(result[field]).all()) for field in ('semantic_logit','delta','residual_logit')),'Nonfinite degraded output')
                        batch[name]={key:result[field].cpu().tolist() for key,field in [('s','semantic_logit'),('d','delta'),('r','residual_logit')]}
                for name,values in batch.items():
                    for key,values_ in values.items():predictions[name][key].extend(values_)
                for j,row in enumerate(val[offset:offset+len(rgb)]):
                    handle.write(json.dumps(dict(evaluation_row_index=offset+j,image_path=row['image_path'],video_id=row['video_id'],family_id=row['family_id'],label=row['label'],
                        source_frame_number=row.get('source_frame_number'),source_frame_number_known='source_frame_number' in row,
                        scores={name:{key:batch[name][key][j] for key in ('s','d','r')} for name in batch}),allow_nan=False)+'\n')
                offset+=len(rgb)
                if offset%1024<len(rgb) or offset==len(val):guard.check();state(run,'degraded_streaming',done=offset,total=len(val),resources=resources(run),full_dataset_cache=False)
            plan.require(not old.read().strip(),'Extra old degraded frame')
        plan.require(offset==len(val),'Incomplete degraded source')
        records=video_records(predictions);records['Semantic']=[dict(row,score=row['semantic']) for row in records['OldV2']]
        degraded=plan.merge_videos(records);dump(evaluation/'degraded_VIDEOS.json',degraded)
        prior_videos={row['video_id']:row for row in plan.read(reuse['videos'])}
        for row in degraded:
            for name in ('OldV2','ReusedV5IR','Semantic'):plan.require(abs(row['scores'][name]-prior_videos[row['video_id']]['scores'][name])<2e-7,'Reused degraded baseline changed')
        state(run,'source_statistics',view='degraded')
        degraded_summary=plan.statistics(degraded,'fixed degraded source VAL, single residual seed; diagnostic not external domain')
        dump(evaluation/'degraded_SUMMARY.json',degraded_summary)
        reproduced={view:{name:abs(plan.auc([row['label'] for row in rows],[row['scores'][name] for row in rows])-summary['aucs'][name])<1e-12
            for name in summary['aucs']} for view,rows,summary in [('clean',clean,clean_summary),('degraded',degraded,degraded_summary)]}
        plan.require(all(all(values.values()) for values in reproduced.values()),'Saved score AUC reproduction failed')
        dump(evaluation/'AUC_REPRODUCTION.json',reproduced)
        report=dict(time=stamp(),source={'clean':clean_summary,'degraded':degraded_summary},seed=plan.SEED,seed_standard_deviation=None,
            pilot_positive_signal=plan.pilot_signal(clean_summary,degraded_summary),semantic_shared=True,source_test_access=False,
            target_inference=False,promotion=False,automatic_expansion=False,semantic_retraining=False,
            original_residual_training_repeated=False,registration_sha256=lock['registration_sha256'])
        guard.check();final_resources=resources(run)
        dump(run/'FINAL_REPORT.json',report);dump(run/'COMPLETE.json',dict(time=stamp(),final_report_sha256=plan.digest(run/'FINAL_REPORT.json')))
        state(run,'completed',pilot_positive_signal=report['pilot_positive_signal'],source=report['source'],resources=final_resources,promotion=False)


if __name__=='__main__':
    try:main()
    except Exception as error:
        run=ROOT/plan.RUN_NAME
        prefix='PREPARE' if '--prepare-only' in sys.argv else 'PREFLIGHT' if '--preflight-only' in sys.argv else 'TRAIN'
        if run.exists() and not (run/(prefix+'_FAILURE.json')).exists():dump(run/(prefix+'_FAILURE.json'),dict(time=stamp(),error=repr(error),history_preserved=True))
        if run.exists():state(run,'failed',stage_kind=prefix,error=repr(error))
        raise
