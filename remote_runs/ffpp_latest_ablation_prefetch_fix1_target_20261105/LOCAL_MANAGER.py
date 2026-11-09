"""Finite five-arm target acceleration; no source mutation or GPU wait job."""
import argparse
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import manage_target_prefetch_20261105 as template
import manage_latest_dual_ablation_20261105 as source
from build_latest_ablation_prefetch_20261105 import runner,verifier

ROOT,BASE,CONTROL,HOST=source.ROOT,source.BASE,source.CONTROL,source.HOST
MOD=ROOT/'code/dual_stream_20261026/latest_ablation_prefetch_20261105'
REUSE=ROOT/'code/dual_stream_20261026/target_prefetch_20261105'
spec=importlib.util.spec_from_file_location('latest_prefetch_plan',MOD/'plan.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
LOCAL,PARENT=BASE/p.RUN,BASE/p.SOURCE_RUN
DOC='docs/130_五组消融有界读取提速接续_20261105.md'
AUTH=source.AUTH+' 最新用户要求尽快完成及提速：仅独立目标侧车2CPU线程有界读取，有限CPU/CUDA等价测速；不改运行源训练，不恢复旧任务/监控，剩余源读取仅草案。'
read,sha,dump,stamp,remote,scp=source.read,source.sha,source.dump,source.stamp,source.remote,source.scp


def scope():
    c=source.scope()
    assert read(PARENT/'REGISTRATION.json')['registration_sha256']==p.SOURCE_REG
    return c


def update(phase,proof,**changes):
    c=scope();item=c['current_round'].get('latest_ablation_acceleration',{}).copy()
    item.update(changes,run=p.RUN,source_run=p.SOURCE_RUN,source_registration_sha256=p.SOURCE_REG,
        training_unchanged=True,workers=2,pending_batches=2,new_retained_updates=0,
        extra_benchmark_forwards=p.EXTRA_BENCHMARK_FORWARDS,document=DOC)
    from update_maskfree_control_20261102 import update as atomic
    atomic(sha(CONTROL),dict(human_authorization=AUTH,latest_ablation_acceleration=item,
        monitor_resume_authorized=False),phase,Path(proof).resolve().relative_to(ROOT).as_posix())
    if changes.get('complete'):
        # Preserve source registration and run; explicitly name the sidecar target.
        source.AUTH=AUTH
        source.update('Latest_dual_ablation_first_complete_verified',proof,
            status='complete_archived_verified',started=False,complete=True,
            target_execution_run=p.RUN,target_execution_registration_sha256=verified()['registration_sha256'],
            target_archive=changes['target_archive'],
            target_verification_sha256=changes['target_verification_sha256'],next_action=changes['next_action'])


def tests():
    q=subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'tools/test_latest_ablation_prefetch_20261105.py')],
        capture_output=True,text=True,encoding='utf-8')
    assert q.returncode==0,q.stdout+q.stderr
    return dict(passed=True,scope='stdlib concurrency/AST/budget/registered SHA only; not CUDA',stdout=q.stdout,stderr=q.stderr)


def verified():
    lock=read(LOCAL/'REGISTRATION.json')
    assert lock['configuration']==p.configuration()
    assert lock['registration_sha256']==p.fingerprint({k:v for k,v in lock.items() if k!='registration_sha256'})
    for name,digest in lock['files'].items():assert sha(LOCAL/'bundle'/name)==digest,name
    assert sha(CONTROL.with_name('ROUNDS.jsonl'))==lock['ledger_sha256']
    return lock


def register():
    scope();assert not LOCAL.exists(),'Preserve preparation and test evidence'
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
    copy(MOD/'plan.py','implementation/plan.py')
    for name in ('prefetch.py','cpu_benchmark.py'):copy(REUSE/name,'implementation/'+name)
    generated('implementation/run_cuda.py',runner((PARENT/'bundle/implementation/run_cuda.py').read_text(encoding='utf-8')))
    generated('LOCAL_VERIFIER.py',verifier((PARENT/'bundle/LOCAL_VERIFIER.py').read_text(encoding='utf-8')))
    for src,name in ((ROOT/DOC,'SCIENTIFIC_PROTOCOL.md'),(Path(__file__),'LOCAL_MANAGER.py'),
        (ROOT/'tools/build_latest_ablation_prefetch_20261105.py','BUNDLE_TRANSFORM_SOURCE.py'),
        (ROOT/'tools/build_target_prefetch_20261105.py','UPSTREAM_PREFETCH_TRANSFORM.py'),
        (ROOT/'tools/manage_target_prefetch_20261105.py','UPSTREAM_PREFETCH_MANAGER.py'),
        (ROOT/'tools/test_latest_ablation_prefetch_20261105.py','STATIC_TEST_SOURCE.py'),
        (ROOT/'tools/test_target_prefetch_20261105.py','UPSTREAM_PREFETCH_TEST_SOURCE.py'),
        (PARENT/'REGISTRATION.json','SOURCE_REGISTRATION.json'),
        (PARENT/'bundle/SCIENTIFIC_PROTOCOL.md','SOURCE_SCIENTIFIC_PROTOCOL.md')):copy(src,name)
    cpu_parent=BASE/'ffpp_latest_ablation_prefetch_target_20261105'
    cpu=read(cpu_parent/'CPU_BENCHMARK.json')
    assert cpu['passed'] and cpu['total_image_reads']==1024 and not cpu['CUDA_used']
    assert len({t['pixel_sha256'] for t in cpu['trials']})==1
    for name in ('prefetch.py','cpu_benchmark.py','checked_pixels.py'):
        assert sha(cpu_parent/'bundle/implementation'/name)==sha(bundle/'implementation'/name)
    for name in ('inputs/source_val.json','SOURCE_REGISTRATION.json'):
        assert sha(cpu_parent/'bundle'/name)==sha(bundle/name)
    copy(cpu_parent/'CPU_BENCHMARK.json','CPU_BENCHMARK_REFERENCE.json')
    copy(cpu_parent/'REGISTRATION.json','CPU_BENCHMARK_REFERENCE_REGISTRATION.json')
    external=read(PARENT/'bundle/EXTERNAL_LOCK.json')
    external[p.REMOTE+'/'+p.SOURCE_RUN+'/REGISTRATION.json']=sha(PARENT/'REGISTRATION.json')
    dump(bundle/'EXTERNAL_LOCK.json',external);dump(bundle/'STATIC_TESTS.json',check)
    lock=dict(configuration=p.configuration(),files={f.relative_to(bundle).as_posix():sha(f) for f in bundle.rglob('*') if f.is_file()},
        entrypoint='implementation/run_cuda.py',ledger_sha256=sha(CONTROL.with_name('ROUNDS.jsonl')))
    lock['registration_sha256']=p.fingerprint(lock)
    dump(bundle/'REGISTRATION.json',lock);dump(LOCAL/'REGISTRATION.json',lock)
    update('Latest_ablation_acceleration_registered',LOCAL/'REGISTRATION.json',status='registered',
        registered=True,uploaded=False,started=False,cuda_preflight_passed=False,registration_sha256=lock['registration_sha256'])
    print(json.dumps(dict(registration_sha256=lock['registration_sha256'],files=len(lock['files']))))


def borrow(name):
    """Use the already tested finite transport/launch, with explicit new globals."""
    body=inspect.getsource(getattr(template,name))
    namespace=dict(globals())
    exec(compile(body,'latest_prefetch_'+name,'exec'),namespace)
    return namespace[name]


def upload():borrow('upload')()
def cpu_benchmark():
    raise ValueError('Fixed 1024 reads already completed. Use adopt-cpu; no repeated benchmark.')


def adopt_cpu():
    lock=verified();assert read(LOCAL/'UPLOAD_VERIFICATION.json')['passed']
    reference=read(LOCAL/'bundle/CPU_BENCHMARK_REFERENCE.json')
    refreg=read(LOCAL/'bundle/CPU_BENCHMARK_REFERENCE_REGISTRATION.json')
    assert reference['registration_sha256']==refreg['registration_sha256']
    assert reference['passed'] and reference['total_image_reads']==1024 and not reference['CUDA_used']
    result=remote(f'''import pathlib,json,hashlib
r=pathlib.Path({p.REMOTE!r})/{p.RUN!r};old=r.parent/'ffpp_latest_ablation_prefetch_target_20261105'
read=lambda q:json.loads(q.read_text());sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
assert sha(old/'CPU_BENCHMARK.json')==sha(r/'CPU_BENCHMARK_REFERENCE.json')
assert sha(old/'REGISTRATION.json')==sha(r/'CPU_BENCHMARK_REFERENCE_REGISTRATION.json')
for name in ['implementation/prefetch.py','implementation/cpu_benchmark.py','implementation/checked_pixels.py','inputs/source_val.json','SOURCE_REGISTRATION.json']:
 assert sha(old/name)==sha(r/name)
value=dict(passed=True,registration_sha256={lock['registration_sha256']!r},
 reference_registration_sha256={refreg['registration_sha256']!r},reference_sha256=sha(r/'CPU_BENCHMARK_REFERENCE.json'),
 speedup={reference['speedup']!r},reference_image_reads=1024,new_image_reads=0,CUDA_used=False,
 same_decoder_and_inputs=True,target_CUDA_pending=True)
with (r/'CPU_BENCHMARK_ADOPTION.json').open('x') as f:json.dump(value,f,indent=2)
print(json.dumps(value))''')
    dump(LOCAL/'CPU_BENCHMARK_ADOPTION.json',result)
    update('Latest_ablation_acceleration_CPU_verified_CUDA_pending',LOCAL/'CPU_BENCHMARK_ADOPTION.json',
        status='CPU_measured_CUDA_pending',cpu_benchmark_passed=True,cpu_speedup=result['speedup'],cuda_preflight_passed=False)
    print(json.dumps(result))
def launch(stage):borrow('launch')(stage)
def status():borrow('status')()
def archive_target():borrow('archive_target')()


def adopt_source():
    lock=verified();assert not (LOCAL/'SOURCE_ADOPTION.json').exists()
    proof=read(PARENT/'SOURCE_LOCAL_VERIFICATION.json')
    assert proof['passed'] and proof['registration_sha256']==p.SOURCE_REG
    assert proof['verified_epochs']==p.ENDPOINTS*10
    sourceinfo=scope()['current_round']['latest_dual_ablation']['batches']['first']
    archived=ROOT/sourceinfo['source_archive']
    assert sha(archived/'LOCAL_VERIFICATION.json')==sha(PARENT/'SOURCE_LOCAL_VERIFICATION.json')==sourceinfo['source_verification_sha256']
    result=remote(f'''import pathlib,json,hashlib,os,subprocess
r=pathlib.Path({p.REMOTE!r})/{p.RUN!r};src=r.parent/{p.SOURCE_RUN!r}
read=lambda q:json.loads(q.read_text());sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
done=read(src/'SOURCE_COMPLETE.json');assert done['passed'] and done['registration_sha256']=={p.SOURCE_REG!r}
assert not (src/'SOURCE_FAILURE.json').exists()
a=read(src/'SOURCE_LAUNCH.json');proc=pathlib.Path('/proc')/str(a['pid'])/'cmdline';assert not proc.exists() or not proc.read_bytes()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
for prefix in ('TARGET_PREFLIGHT','TARGET'):
 assert not any((src/(prefix+s)).exists() for s in ('_LAUNCH.json','_STARTED.json','_COMPLETE.json','_FAILURE.json')),'Do not duplicate original target'
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
    update('Latest_ablation_acceleration_source_adopted',LOCAL/'SOURCE_ADOPTION.json',status='source_verified_adopted',source_adopted=True)
    print(json.dumps(result))


def verify_preflight():
    lock=verified();cfg=p.configuration()
    expected=dict(source_reload_probe=p.ENDPOINTS*64,target_repeat_probe=p.ENDPOINTS*224,
        acceleration_benchmark=p.EXTRA_BENCHMARK_FORWARDS)
    result=remote(f'''import pathlib,json,subprocess
r=pathlib.Path({p.REMOTE!r})/{p.RUN!r};read=lambda q:json.loads(q.read_text())
assert not (r/'TARGET_PREFLIGHT_FAILURE.json').exists()
done=read(r/'TARGET_PREFLIGHT_COMPLETE.json');assert done['passed'] and done['registration_sha256']=={lock['registration_sha256']!r}
assert done['preflight_forwards']=={cfg['target_preflight_forwards_per_invocation']} and done['new_optimizer_updates']==0
assert done['network_frame_forwards']=={expected!r}
choice=read(r/'ACCELERATION_DECISION.json');assert choice['passed'] and choice['max_output_error']<=2e-7
assert choice['enable_prefetch']==(choice['speedup']>=1.05)
assert choice['source_registration_sha256']=={p.SOURCE_REG!r}
assert choice['selection_lock_sha256']==read(r/'SOURCE_ADOPTION.json')['selection_lock_sha256']
a=read(r/'TARGET_PREFLIGHT_LAUNCH.json');proc=pathlib.Path('/proc')/str(a['pid'])/'cmdline';assert not proc.exists() or not proc.read_bytes()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
print(json.dumps(dict(passed=True,process_exited=True,registration_sha256=done['registration_sha256'],complete=done,decision=choice)))''')
    dump(LOCAL/'TARGET_PREFLIGHT_EXIT_VERIFICATION.json',result)
    update('Latest_ablation_acceleration_CUDA_verified_exited',LOCAL/'TARGET_PREFLIGHT_EXIT_VERIFICATION.json',
        status='CUDA_verified_exited',cuda_preflight_passed=True,enable_prefetch=result['decision']['enable_prefetch'],
        speedup=result['decision']['speedup'],started=False)
    print(json.dumps(result))


def verify_source(path):
    assert sha(ROOT/'tools/verify_latest_dual_ablation_20261105.py')==sha(PARENT/'bundle/LOCAL_VERIFIER.py')
    source.AUTH=AUTH
    source.verify_archive('source',path)


def verify_target(path):
    lock=verified();assert sha(LOCAL/'bundle/LOCAL_VERIFIER.py')==lock['files']['LOCAL_VERIFIER.py']
    spec=importlib.util.spec_from_file_location('frozen_latest_target_verifier',LOCAL/'bundle/LOCAL_VERIFIER.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.m=sys.modules[__name__]
    result=module.verify_target(Path(path).resolve())
    delivery=read(LOCAL/'FINAL_DELIVERY.json')
    assert delivery['passed'] and delivery['seeds']==[2027]
    assert not (PARENT/'FINAL_DELIVERY.json').exists(),'Never overwrite prior delivery'
    delivery.update(source_run=p.SOURCE_RUN,source_registration_sha256=p.SOURCE_REG,target_execution_run=p.RUN)
    dump(PARENT/'FINAL_DELIVERY.json',delivery)
    print(json.dumps(result))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('tests','register','upload','cpu-benchmark','adopt-cpu',
        'archive-source','verify-source','adopt-source','target-preflight','verify-preflight','target','status','archive-target','verify-target'))
    ap.add_argument('--archive');args=ap.parse_args()
    if args.action in ('target-preflight','target'):launch(args.action)
    elif args.action in ('verify-source','verify-target'):globals()[args.action.replace('-','_')](args.archive)
    elif args.action=='archive-source':source.archive('SOURCE')
    else:
        value=globals()[args.action.replace('-','_')]()
        if value is not None:print(json.dumps(value))
