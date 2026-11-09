"""Shared remote-only runtime; neural imports are delayed until owned CUDA."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import confirmation_plan as q


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
        if hasattr(os,'posix_fadvise'):os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    return h.hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(path,x):
    with Path(path).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def resources(run,additional=0,disk=0):
    cg=Path('/sys/fs/cgroup/memory');cap=int((cg/'memory.limit_in_bytes').read_text());used=int((cg/'memory.usage_in_bytes').read_text());free=shutil.disk_usage(run).free
    assert cap-used>=6*1024**3+additional,'6GiB cgroup reserve';assert free>=1024**3+disk,'1GiB disk reserve'
    return dict(memory_cap=cap,memory_used=used,disk_free=free)
def state(run,stage,**kw):
    item=dict(time=stamp(),stage=stage,resources=resources(run),**kw)
    with (run/'.STATE.pending').open('w') as f:json.dump(item,f,allow_nan=False)
    os.replace(run/'.STATE.pending',run/'STATE.json')
    with (run/'STAGES.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
    print(json.dumps(item),flush=True)
def verify(run):
    assert run==Path(q.REMOTE)/q.RUN
    lock=read(run/'REGISTRATION.json');assert lock['configuration']==q.configuration()
    assert lock['registration_sha256']==q.fingerprint({k:v for k,v in lock.items() if k!='registration_sha256'})
    for name,value in lock['files'].items():assert sha(run/name)==value,('Bundle changed',name)
    for name,value in read(run/'EXTERNAL_LOCK.json').items():assert sha(name)==value,('Dependency changed',name)
    assert read(run/'SELECTION.json')['winner']=='SemanticHalf'
    from crosslayer_plan import verify_split
    train,val=(read(run/'inputs'/f'{name}.json') for name in ('fit','source_val'))
    assert verify_split(train,val)==read(run/'SPLIT_PROOF.json')
    assert (len(train),len({r['video_id'] for r in train}),len({r['family_id'] for r in train}))==(11492,2873,291)
    assert (len(val),len({r['video_id'] for r in val}),len({r['family_id'] for r in val}))==(22080,690,70)
    config=read(run/'V5_CONFIG.json')
    assert tuple(config[k] for k in ('batch_size','workers','epochs','dual_lr','weight_decay','residual_aux_weight','consistency_weight'))==(16,2,10,.0003,.01,.1,.05)
    return lock
def tensor_hash(values):
    h=hashlib.sha256()
    for name,t in sorted(values.items()):h.update(name.encode());h.update(t.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def environment(run,guard):
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.cuda.init();x=torch.empty(1,device='cuda');torch.cuda.synchronize();guard.check();del x
    sys.path.insert(0,str(run/'implementation/v5'));sys.path.append(q.REMOTE)
    import gpu_pipeline as pipeline
    import full_source_v2 as compact
    import joint_search_v4b as joint
    from crosslayer_cuda import CrossLayerDual,encode
    from multiscale_cuda import upgrade
    def cuda_load(path):
        obj=torch.load(path,map_location='cuda',weights_only=True)
        return obj,obj.get('ema_state') or obj['model_state']
    pipeline.state_of=cuda_load
    config=read(run/'V5_CONFIG.json')
    with torch.device('cuda'):anchor=compact.CompactSemantic(config['semantic_checkpoint']).cuda().eval()
    spec={**config['model'],**joint.ARCHITECTURES['IR_LayerScale']}
    def make(seed):
        pipeline.seed_all(seed)
        # Preserve the historical constructor/RNG placement exactly; no forward or checkpoint load on CPU.
        model=CrossLayerDual(upgrade(joint.JointDual(joint.SemanticExit(anchor),spec).cuda(),'ShallowMultiScale'),'CrossLayerNoImage').cuda()
        assert all(t.is_cuda for t in model.parameters());return model
    def encoder():
        from safetensors import safe_open
        # Preserve from_pretrained attention/config selection, but construct/load all parameters on CUDA without accelerate.
        with safe_open(str(Path(config['clip_path'])/'model.safetensors'),framework='pt',device=0,backend='pread') as handle:
            tensors={key:handle.get_tensor(key) for key in handle.keys()}
        assert all(t.is_cuda for t in tensors.values())
        with torch.device('cuda'):
            value=torch.nn.Module()
            value.clip=pipeline.CLIPVisionModel.from_pretrained(config['clip_path'],local_files_only=True,state_dict=tensors).cuda().eval().requires_grad_(False)
            value.anchor=pipeline.SemanticTokens(config['semantic_checkpoint']).cuda().eval()
        del tensors
        assert all(t.is_cuda and not t.requires_grad for t in value.parameters());guard.check();return value
    def load(seed,path,registration,expected_hash=None):
        saved=torch.load(path,map_location='cuda',weights_only=True)
        assert [saved[k] for k in ('arm','epoch','seed','registration_sha256')]==['CrossLayerNoImage',10,seed,registration]
        steps=[float(v['step']) for v in saved['optimizer']['state'].values() if 'step' in v];assert steps and all(x==7190 for x in steps)
        assert all(t.is_cuda for t in saved['model'].values())
        digest=tensor_hash(saved['model'])
        if expected_hash:assert digest==expected_hash
        model=make(seed);model.load_state_dict(saved['model'],strict=True);model.eval().requires_grad_(False)
        assert tensor_hash(model.state_dict())==digest
        del saved;return model,digest
    return torch,pipeline,joint,make,encoder,encode,load,spec
