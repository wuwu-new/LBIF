"""Explicit target-only acceleration; never restart source or create monitoring."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import manage_patch_region_full_20261105 as source
from build_target_prefetch_20261105 import runner, verifier

ROOT,BASE,CONTROL,HOST=source.ROOT,source.BASE,source.CONTROL,source.HOST
MOD=ROOT/'code/dual_stream_20261026/target_prefetch_20261105'
spec=importlib.util.spec_from_file_location('prefetch_target_plan',MOD/'plan.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
LOCAL,PARENT=BASE/p.RUN,BASE/p.SOURCE_RUN
DOC='docs/127_不改运行训练的有界读取提速协议_20261105.md'
AUTH=source.AUTH+' 最新用户“按照你的建议帮我提速”：仅独立目标侧车2CPU线程有界读取；原源训练不改/不中止/不重跑，固定等价性与ABBA有限测速后决定是否启用，不恢复旧任务或监控。'
read,sha,dump,stamp,remote,scp=source.read,source.sha,source.dump,source.stamp,source.remote,source.scp


def scope():
    c=source.scope()
    assert read(PARENT/'REGISTRATION.json')['registration_sha256']==p.SOURCE_REG
    return c


def update(phase,proof,**changes):
    c=scope();item=c['current_round'].get('target_acceleration',{}).copy()
    item.update(changes,run=p.RUN,source_run=p.SOURCE_RUN,source_registration_sha256=p.SOURCE_REG,
        training_unchanged=True,workers=2,pending_batches=2,new_retained_updates=0,
        extra_benchmark_forwards=6144,document=DOC)
    from update_maskfree_control_20261102 import update as atomic
    atomic(sha(CONTROL),dict(human_authorization=AUTH,target_acceleration=item,
        monitor_resume_authorized=False,next_action=item.get('next_action',
        'Source keeps running unchanged. Target-only sidecar after source exits and is archived/verified; no restart or monitor.')),
        phase,Path(proof).resolve().relative_to(ROOT).as_posix())


def tests():
    q=subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'tools/test_target_prefetch_20261105.py')],
        capture_output=True,text=True,encoding='utf-8')
    assert q.returncode==0,q.stdout+q.stderr
    return dict(passed=True,scope='stdlib concurrency/AST/protocol only; not CUDA or measured acceleration',stdout=q.stdout,stderr=q.stderr)


def verified():
    lock=read(LOCAL/'REGISTRATION.json')
    assert lock['configuration']==p.configuration()
    assert lock['registration_sha256']==p.fingerprint({k:v for k,v in lock.items() if k!='registration_sha256'})
    for name,digest in lock['files'].items():assert sha(LOCAL/'bundle'/name)==digest,name
    assert sha(CONTROL.with_name('ROUNDS.jsonl'))==lock['ledger_sha256']
    return lock


def register():
    scope();assert not LOCAL.exists(),'Preserve existing preparation'
    check=tests();old=read(PARENT/'REGISTRATION.json')
    assert old['registration_sha256']==p.fingerprint({k:v for k,v in old.items() if k!='registration_sha256'})
    for name,digest in old['files'].items():assert sha(PARENT/'bundle'/name)==digest,name
    bundle=LOCAL/'bundle';bundle.mkdir(parents=True)
    skip={'implementation/plan.py','implementation/run_cuda.py','LOCAL_MANAGER.py','LOCAL_VERIFIER.py',
        'SCIENTIFIC_PROTOCOL.md','STATIC_TEST_SOURCE.py','STATIC_TESTS.json','EXTERNAL_LOCK.json','BUNDLE_TRANSFORM_SOURCE.py'}
    def copy(src,name):
        dst=bundle/name;dst.parent.mkdir(parents=True,exist_ok=True);assert not dst.exists();shutil.copyfile(src,dst)
    def generated(name,text):
        dst=bundle/name;dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
    for name in old['files']:
        if name not in skip:copy(PARENT/'bundle'/name,name)
    copy(PARENT/'bundle/implementation/plan.py','implementation/source_plan.py')
    for name in ('plan.py','prefetch.py','cpu_benchmark.py'):copy(MOD/name,'implementation/'+name)
    generated('implementation/run_cuda.py',runner((PARENT/'bundle/implementation/run_cuda.py').read_text(encoding='utf-8')))
    generated('LOCAL_VERIFIER.py',verifier((PARENT/'bundle/LOCAL_VERIFIER.py').read_text(encoding='utf-8')))
    for src,name in ((ROOT/DOC,'SCIENTIFIC_PROTOCOL.md'),(Path(__file__),'LOCAL_MANAGER.py'),
        (ROOT/'tools/build_target_prefetch_20261105.py','BUNDLE_TRANSFORM_SOURCE.py'),
        (ROOT/'tools/test_target_prefetch_20261105.py','STATIC_TEST_SOURCE.py'),
        (PARENT/'REGISTRATION.json','SOURCE_REGISTRATION.json'),
        (PARENT/'bundle/SCIENTIFIC_PROTOCOL.md','SOURCE_SCIENTIFIC_PROTOCOL.md')):copy(src,name)
    external=read(PARENT/'bundle/EXTERNAL_LOCK.json')
    external[p.REMOTE+'/'+p.SOURCE_RUN+'/REGISTRATION.json']=sha(PARENT/'REGISTRATION.json')
    dump(bundle/'EXTERNAL_LOCK.json',external);dump(bundle/'STATIC_TESTS.json',check)
    lock=dict(configuration=p.configuration(),files={f.relative_to(bundle).as_posix():sha(f) for f in bundle.rglob('*') if f.is_file()},
        entrypoint='implementation/run_cuda.py',ledger_sha256=sha(CONTROL.with_name('ROUNDS.jsonl')))
    lock['registration_sha256']=p.fingerprint(lock)
    dump(bundle/'REGISTRATION.json',lock);dump(LOCAL/'REGISTRATION.json',lock)
    update('Target_acceleration_registered_training_untouched',LOCAL/'REGISTRATION.json',status='registered',
        registered=True,uploaded=False,started=False,cuda_preflight_passed=False,registration_sha256=lock['registration_sha256'])
    print(json.dumps(dict(registration_sha256=lock['registration_sha256'],files=len(lock['files']))))


def upload():
    # Reuse the independently SHA-checked transport only; its old globals are not mutated.
    import inspect
    body=inspect.getsource(source.transport.upload)
    namespace=dict(vars(source.transport),p=p,LOCAL=LOCAL,PARENT=PARENT,scope=scope,verified=verified,update=update)
    exec(compile(body,'prefetch_readonly_transport','exec'),namespace)
    namespace['upload']()


def cpu_benchmark():
    lock=verified();assert read(LOCAL/'UPLOAD_VERIFICATION.json')['passed']
    result=remote(f'''import pathlib,json,subprocess,os,shutil
r=pathlib.Path({p.REMOTE!r})/{p.RUN!r};cg=pathlib.Path('/sys/fs/cgroup/memory')
assert int((cg/'memory.limit_in_bytes').read_text())-int((cg/'memory.usage_in_bytes').read_text())>7*1024**3
assert shutil.disk_usage(r).free>3*1024**3
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
q=subprocess.run(['/root/miniconda3/bin/python','-B',str(r/'implementation/cpu_benchmark.py')],cwd=r,env=env,capture_output=True,text=True,timeout=120)
assert q.returncode==0,q.stdout+q.stderr
out=json.loads(q.stdout);assert out['passed'] and out['registration_sha256']=={lock['registration_sha256']!r}
print(json.dumps(out))''',timeout=150)
    dump(LOCAL/'CPU_BENCHMARK.json',result)
    update('Target_acceleration_CPU_measured_CUDA_pending',LOCAL/'CPU_BENCHMARK.json',status='CPU_measured_CUDA_pending',
        cpu_benchmark_passed=True,cpu_speedup=result['speedup'],cuda_preflight_passed=False)
    print(json.dumps(result))


def adopt_source():
    lock=verified();assert not (LOCAL/'SOURCE_ADOPTION.json').exists()
    proof=read(PARENT/'SOURCE_LOCAL_VERIFICATION.json')
    assert proof['passed'] and proof['registration_sha256']==p.SOURCE_REG and proof['all30_epochs_verified']
    source_info=scope()['current_round']['patch_region_full']
    archived=ROOT/source_info['source_archive']
    assert sha(archived/'LOCAL_VERIFICATION.json')==sha(PARENT/'SOURCE_LOCAL_VERIFICATION.json')
    assert source_info['source_verification_sha256']==sha(archived/'LOCAL_VERIFICATION.json')
    result=remote(f'''import pathlib,json,hashlib,os,subprocess
r=pathlib.Path({p.REMOTE!r})/{p.RUN!r};src=r.parent/{p.SOURCE_RUN!r}
read=lambda q:json.loads(q.read_text());sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
done=read(src/'SOURCE_COMPLETE.json');assert done['passed'] and done['registration_sha256']=={p.SOURCE_REG!r}
assert not (src/'SOURCE_FAILURE.json').exists()
a=read(src/'SOURCE_LAUNCH.json');proc=pathlib.Path('/proc')/str(a['pid'])/'cmdline';assert not proc.exists() or not proc.read_bytes()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
for prefix in ('TARGET_PREFLIGHT','TARGET'):
 assert not any((src/(prefix+s)).exists() for s in ('_LAUNCH.json','_STARTED.json','_COMPLETE.json','_FAILURE.json')),'Original target already touched'
value={proof!r};proofpath=src/'SOURCE_LOCAL_VERIFICATION.json'
if proofpath.exists():assert read(proofpath)==value
else:
 with proofpath.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
files={{}}
for name in ['SOURCE_COMPLETE.json','SOURCE_LAUNCH.json','SELECTED_ENDPOINT_LOCK.json','SOURCE_LOCAL_VERIFICATION.json']:
 assert not (r/name).exists();os.link(src/name,r/name);files[name]=sha(r/name)
assert files['SELECTED_ENDPOINT_LOCK.json']==done['selection_lock_sha256']
out=dict(source_run={p.SOURCE_RUN!r},source_registration_sha256={p.SOURCE_REG!r},
 sidecar_registration_sha256={lock['registration_sha256']!r},files=files,selection_lock_sha256=done['selection_lock_sha256'],
 original_source_archive_verification_sha256={sha(archived/'LOCAL_VERIFICATION.json')!r},source_exited=True,new_training=False)
with (r/'SOURCE_ADOPTION.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out))''')
    dump(LOCAL/'SOURCE_ADOPTION.json',result)
    update('Target_acceleration_source_adopted',LOCAL/'SOURCE_ADOPTION.json',status='source_verified_adopted',source_adopted=True)
    print(json.dumps(result))


def launch(stage):
    assert stage in ('target-preflight','target')
    scope();lock=verified();assert read(LOCAL/'UPLOAD_VERIFICATION.json')['passed']
    assert read(LOCAL/'SOURCE_ADOPTION.json')['source_exited']
    if stage=='target':assert read(LOCAL/'TARGET_PREFLIGHT_EXIT_VERIFICATION.json')['passed']
    result=remote(f'''import pathlib,json,subprocess,os,shutil,datetime
r=pathlib.Path({p.REMOTE!r})/{p.RUN!r};src=r.parent/{p.SOURCE_RUN!r};prefix={stage.upper().replace('-','_')!r}
read=lambda q:json.loads(q.read_text())
assert read(r/'REGISTRATION.json')['registration_sha256']=={lock['registration_sha256']!r}
for suffix in ['_LAUNCH.json','_STARTED.json','_COMPLETE.json','_FAILURE.json','.log']:assert not (r/(prefix+suffix)).exists()
a=read(src/'SOURCE_LAUNCH.json');proc=pathlib.Path('/proc')/str(a['pid'])/'cmdline';assert not proc.exists() or not proc.read_bytes()
if {stage!r}=='target':
 a=read(r/'TARGET_PREFLIGHT_LAUNCH.json');proc=pathlib.Path('/proc')/str(a['pid'])/'cmdline';assert not proc.exists() or not proc.read_bytes()
 assert read(r/'TARGET_PREFLIGHT_COMPLETE.json')['passed']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy; no queue'
cg=pathlib.Path('/sys/fs/cgroup/memory');assert int((cg/'memory.limit_in_bytes').read_text())-int((cg/'memory.usage_in_bytes').read_text())>8*1024**3
assert shutil.disk_usage(r).free>3*1024**3
cmd=['/root/miniconda3/bin/python','-u','-B',str(r/'implementation/run_cuda.py'),'--run',str(r),'--stage',{stage!r}]
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
with (r/(prefix+'.log')).open('xb') as f:child=subprocess.Popen(cmd,cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
out=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=child.pid,registration_sha256={lock['registration_sha256']!r},command=cmd)
with (r/(prefix+'_LAUNCH.json')).open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out))''')
    path=LOCAL/(stage.upper().replace('-','_')+'_LAUNCH.json');dump(path,result)
    update('Target_acceleration_'+stage+'_launched',path,status=stage+'_launched',started=True)
    print(json.dumps(result))


def verify_preflight():
    lock=verified()
    result=remote(f'''import pathlib,json,subprocess
r=pathlib.Path({p.REMOTE!r})/{p.RUN!r};read=lambda q:json.loads(q.read_text())
assert not (r/'TARGET_PREFLIGHT_FAILURE.json').exists()
done=read(r/'TARGET_PREFLIGHT_COMPLETE.json');assert done['passed'] and done['registration_sha256']=={lock['registration_sha256']!r}
assert done['preflight_forwards']==7008 and done['new_optimizer_updates']==0
assert done['network_frame_forwards']==dict(source_reload_probe=192,target_repeat_probe=672,acceleration_benchmark=6144)
choice=read(r/'ACCELERATION_DECISION.json');assert choice['passed'] and choice['max_output_error']<=2e-7
assert choice['enable_prefetch']==(choice['speedup']>=1.05)
a=read(r/'TARGET_PREFLIGHT_LAUNCH.json');proc=pathlib.Path('/proc')/str(a['pid'])/'cmdline';assert not proc.exists() or not proc.read_bytes()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
print(json.dumps(dict(passed=True,process_exited=True,registration_sha256=done['registration_sha256'],complete=done,decision=choice)))''')
    dump(LOCAL/'TARGET_PREFLIGHT_EXIT_VERIFICATION.json',result)
    update('Target_acceleration_CUDA_verified_exited',LOCAL/'TARGET_PREFLIGHT_EXIT_VERIFICATION.json',status='CUDA_verified_exited',
        cuda_preflight_passed=True,enable_prefetch=result['decision']['enable_prefetch'],speedup=result['decision']['speedup'],started=False)
    print(json.dumps(result))


def status():
    result=remote(f'''import pathlib,json,subprocess,shutil,datetime
root=pathlib.Path({p.REMOTE!r});out=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),GPU=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],text=True),disk_free=shutil.disk_usage(root).free)
for name in [{p.SOURCE_RUN!r},{p.RUN!r}]:
 r=root/name;item={{}}
 for file in ['STATE.json','SOURCE_COMPLETE.json','SOURCE_FAILURE.json','TARGET_PREFLIGHT_COMPLETE.json','TARGET_PREFLIGHT_FAILURE.json','TARGET_COMPLETE.json','TARGET_FAILURE.json','ACCELERATION_DECISION.json']:
  if (r/file).exists():item[file]=json.loads((r/file).read_text())
 for prefix in ['SOURCE','TARGET_PREFLIGHT','TARGET']:
  if (r/(prefix+'_LAUNCH.json')).exists():
   a=json.loads((r/(prefix+'_LAUNCH.json')).read_text());proc=pathlib.Path('/proc')/str(a['pid'])/'cmdline';item[prefix+'_alive']=proc.exists() and bool(proc.read_bytes())
 out[name]=item
print(json.dumps(out))''')
    path=LOCAL/('STATUS_'+stamp()+'.json');dump(path,result)
    print(json.dumps(result))


def archive_target():
    import inspect
    body=inspect.getsource(source.base.archive).replace("tarfile.open(path,'x:gz')","tarfile.open(path,'x:gz',dereference=True)")
    namespace=dict(vars(source.base),p=p,LOCAL=LOCAL,scope=scope,verified=verified)
    exec(compile(body,'prefetch_materialized_archive','exec'),namespace)
    namespace['archive']('TARGET')


def verify_target(path):
    spec=importlib.util.spec_from_file_location('prefetch_saved_score_verifier',LOCAL/'bundle/LOCAL_VERIFIER.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.verify_target(Path(path).resolve());print(json.dumps(result))


def verify_source(path):
    # Use the original frozen source verifier, with the latest authorization retained.
    source.AUTH=AUTH
    source.verify_archive('source',path)
    proof=read(PARENT/'SOURCE_LOCAL_VERIFICATION.json')
    update('Target_acceleration_source_archive_verified',PARENT/'SOURCE_LOCAL_VERIFICATION.json',
        status='source_verified_pending_adoption',
        next_action='Adopt verified original source identity into independent target sidecar, run CUDA preflight once, verify exit, then complete seven domains.')
    assert proof['passed'] and proof['registration_sha256']==p.SOURCE_REG


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('tests','register','upload','cpu-benchmark','archive-source','verify-source','adopt-source','target-preflight','verify-preflight','target','status','archive-target','verify-target'))
    parser.add_argument('--archive');args=parser.parse_args()
    if args.action in ('target-preflight','target'):launch(args.action)
    elif args.action in ('verify-target','verify-source'):globals()[args.action.replace('-','_')](args.archive)
    elif args.action=='archive-source':source.archive('SOURCE')
    else:
        result=globals()[args.action.replace('-','_')]()
        if result is not None:print(json.dumps(result))
