import datetime,hashlib,json,os,shutil
from pathlib import Path

def read(path):
    return json.loads(Path(path).read_text())

def dump(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
        if hasattr(os,'posix_fadvise'):os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    return h.hexdigest()

def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def resources(run,additional_memory=0):
    cg=Path('/sys/fs/cgroup/memory');cap=int((cg/'memory.limit_in_bytes').read_text());used=int((cg/'memory.usage_in_bytes').read_text())
    disk=shutil.disk_usage(run).free
    assert cap-used>6*1024**3+additional_memory,'6GiB memory reserve'
    assert disk>1024**3,'1GiB disk reserve'
    return dict(memory_cap=cap,memory_used=used,disk_free=disk)

def state(run,stage,**kw):
    value=dict(time=stamp(),stage=stage,resources=resources(run),**kw)
    with (run/'.STATE.pending').open('w') as f:json.dump(value,f,allow_nan=False)
    os.replace(run/'.STATE.pending',run/'STATE.json')
    with (run/'STAGES.jsonl').open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')
    print(json.dumps(value),flush=True)

def tensor_sha(values):
    h=hashlib.sha256()
    for name,x in sorted(values.items()):
        h.update(name.encode());h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode())
        h.update(x.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()
