"""Complete seven domains for the two new selected-model seeds, no training."""
import argparse
import csv
import fcntl
import gc
import json
from pathlib import Path
import time
import traceback
import confirmation_plan as q
from common_cuda import read,sha,dump,stamp,resources,state,verify,environment,tensor_hash


def run_target(run,preflight):
    prefix='TARGET_PREFLIGHT' if preflight else 'TARGET_RUN'
    assert not any((run/name).exists() for name in (prefix+'_STARTED.json',prefix+'_FAILURE.json','TARGET_COMPLETE.json'))
    with (Path(q.REMOTE)/'.gpu_exclusive.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from gpu_identity_guard import OwnedCudaGuard
        guard=OwnedCudaGuard(on_bind=lambda x:dump(run/(prefix+'_CUDA_OWNER.json'),x));guard.check()
        lock=verify(run);resources(run,2*1024**3,512*1024**2)
        assert read(run/'SOURCE_COMPLETE.json')['report_sha256']==sha(run/'SOURCE_REPORT.json')
        proof=read(run/'SOURCE_LOCAL_VERIFICATION.json');assert proof['passed'] and proof['registration_sha256']==lock['registration_sha256']
        assert proof['source_report_sha256']==sha(run/'SOURCE_REPORT.json')
        proc=Path('/proc')/str(read(run/'SOURCE_TRAIN_LAUNCH.json')['pid'])/'cmdline';assert not proc.exists() or not proc.read_bytes()
        if not preflight:
            assert read(run/'TARGET_PREFLIGHT.json')['passed']
            proc=Path('/proc')/str(read(run/'TARGET_PREFLIGHT_LAUNCH.json')['pid'])/'cmdline';assert not proc.exists() or not proc.read_bytes()
        dump(run/(prefix+'_STARTED.json'),dict(time=stamp(),registration_sha256=lock['registration_sha256']))
        torch,pipeline,joint,make,build_encoder,encode,load,spec=environment(run,guard)
        started=time.monotonic();models={};hashes={};identities={}
        old=read(run/'OLD2027.json');models[2027],hashes[2027]=load(2027,old['path'],q.PARENT_REG,old['model_tensor_sha256'])
        identities[2027]=['CrossLayerNoImage',10,2027,q.PARENT_REG]
        source=read(run/'SOURCE_REPORT.json')
        for seed in q.NEW_SEEDS:
            receipt=source['seed_receipts'][str(seed)];path=run/'seeds'/str(seed)/receipt['checkpoint']
            assert sha(path)==receipt['checkpoint_sha256']
            models[seed],hashes[seed]=load(seed,path,lock['registration_sha256'],receipt['model_tensor_sha256'])
            identities[seed]=receipt['identity']
        encoder=build_encoder();pixels=read(Path(q.REMOTE)/q.HISTORY/'TARGET_PIXEL_SHA256.json');references=read(run/'TARGET_REFERENCE.json')
        original=pipeline.Pixels
        class ReleasedPixels(original):
            def __init__(self,rows):super().__init__([dict(row,sha256=pixels[row['image_path']]) for row in rows],False)
            def __getitem__(self,index):
                value=super().__getitem__(index)
                import os
                with Path(self.rows[index]['image_path']).open('rb') as f:os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
                return value
        errors={};finite={}
        for domain in q.DOMAINS:
            rows=read(run/'manifests'/(domain+'.json'))[:16];dataset=ReleasedPixels(rows);rgb=torch.stack([dataset[i] for i in range(16)])
            with torch.no_grad():tokens=encode(encoder,rgb)
            xx=rgb.cuda().requires_grad_(True)
            with Path(references[domain]['baseline_frames']).open() as f:oldrows=[json.loads(next(f)) for _ in range(16)]
            errors[domain]=0.;finite[domain]={}
            for seed,model in models.items():
                # Parameters/tokens frozen. Any differentiable RGB use would make logit require grad.
                with torch.enable_grad():result=model(xx,tokens)
                assert not result['logit'].requires_grad and bool(torch.isfinite(result['logit']).all())
                finite[domain][str(seed)]=True
                if seed==2027:
                    expected=torch.tensor([q.f32(row['scores']['StrongStatic']['s']+row['scores']['StrongStatic']['d']) for row in oldrows],device='cuda').sigmoid()
                    errors[domain]=float((result['logit'].sigmoid()-expected).abs().max());assert errors[domain]<=2e-7
            guard.check();resources(run)
        proof=dict(time=stamp(),passed=True,registration_sha256=lock['registration_sha256'],models_cuda_only=True,
            owned_nvml_pid=guard.owned_nvml_pid,checkpoint_identities=identities,model_tensor_sha256=hashes,
            old2027_target_probability_errors=errors,finite_and_RGB_independent=finite,model_frame_forwards=336,
            target_encoding_probe_frames=112,optimizer_updates=0,source_TEST=False,target_masks=False,resources=resources(run))
        dump(run/('TARGET_PREFLIGHT.json' if preflight else 'TARGET_RUNTIME_PREFLIGHT.json'),proof)
        if preflight:state(run,'target_preflight_passed');return
        del models[2027];gc.collect();torch.cuda.empty_cache()
        evaluation=run/'target';evaluation.mkdir();counts={}
        from torch.utils.data import DataLoader
        for di,domain in enumerate(q.DOMAINS):
            rows=read(run/'manifests'/(domain+'.json'));real,fake=q.COUNTS[di];assert len(rows)==32*(real+fake)
            output=evaluation/(domain+'_FRAMES.csv');offset=0
            with output.open('x',newline='') as f,Path(references[domain]['baseline_frames']).open() as prior:
                writer=csv.writer(f);writer.writerow(['index','seed3407','seed4099'])
                for rgb in DataLoader(ReleasedPixels(rows),batch_size=16,num_workers=2,pin_memory=True):
                    with torch.no_grad():
                        tokens=encode(encoder,rgb);xx=rgb.cuda();values={seed:models[seed](xx,tokens)['logit'].cpu().tolist() for seed in q.NEW_SEEDS}
                    for j in range(len(rgb)):
                        row=rows[offset+j];saved=json.loads(next(prior));assert saved['evaluation_row_index']==offset+j
                        assert all(saved[k]==row[k] for k in ('image_path','video_id','family_id','label'))
                        assert saved['source_frame_number']==row.get('source_frame_number') and saved['source_frame_number_known']==('source_frame_number' in row)
                        assert all(__import__('math').isfinite(values[seed][j]) for seed in q.NEW_SEEDS)
                        writer.writerow([offset+j,*[values[seed][j] for seed in q.NEW_SEEDS]])
                    offset+=len(rgb)
                    if offset%2048<len(rgb) or offset==len(rows):f.flush();guard.check();state(run,'target_streaming',domain=domain,done=offset,total=len(rows),new_seeds=list(q.NEW_SEEDS))
                assert not prior.read().strip()
            assert offset==len(rows);counts[domain]=offset
            dump(evaluation/(domain+'_COMPLETE.json'),dict(frames=offset,model_frame_forwards=offset*2,pixel_SHA_checked=offset,sha256=sha(output)))
        assert sum(counts.values())==223136
        assert all(tensor_hash(models[seed].state_dict())==hashes[seed] for seed in q.NEW_SEEDS)
        report=dict(time=stamp(),registration_sha256=lock['registration_sha256'],frames_by_domain=counts,
            full_model_frame_forwards=446272,preflight_model_frame_forwards_this_process=336,
            source_TEST=False,target_masks=False,new_training_updates=0,model_weights_unchanged=True,
            seconds=time.monotonic()-started,resources=resources(run))
        dump(run/'TARGET_REPORT.json',report);dump(run/'TARGET_COMPLETE.json',dict(time=stamp(),registration_sha256=lock['registration_sha256'],report_sha256=sha(run/'TARGET_REPORT.json')))
        state(run,'target_complete');del models,encoder;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    try:run_target(args.run.resolve(),args.preflight_only)
    except Exception:
        path=args.run/(('TARGET_PREFLIGHT' if args.preflight_only else 'TARGET_RUN')+'_FAILURE.json')
        if not path.exists():dump(path,dict(time=stamp(),traceback=traceback.format_exc()))
        raise
