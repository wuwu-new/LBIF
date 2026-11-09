"""Explicit bounded latest-model ablations; no monitoring automation."""
import argparse
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import manage_paper_components_remaining_20261104 as transport
from build_latest_dual_ablation_20261105 import model, patch_config, runner

base = transport.base
ROOT, BASE, CONTROL, HOST = base.ROOT, base.BASE, base.CONTROL, base.HOST
MOD = ROOT/'code/dual_stream_20261026/latest_dual_ablation_20261105'
spec = importlib.util.spec_from_file_location('latest_dual_ablation_plan', MOD/'plan.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
PARENT = BASE/p.PARENT
SOURCE_FINAL = PARENT/'source_final_20261105T035550185121Z'
TARGET_FINAL = BASE/p.REFERENCE_TARGET/'target_final_20261105T045623473622Z'
DOC = 'docs/129_最新版双流五项匹配消融补齐协议_20261105.md'
AUTH = '用户2026-11-05明确基于最新版不退化语义，并授权按建议补齐五组匹配消融；保留现有语义/读出/融合，复用完整模型及语义，先2027再3407/4099，各五组10epoch源VAL选模及全七域。非可选边界损失组，无新CLIP、无旧任务/监控恢复或按效果扩预算。'
read, sha, dump, stamp, remote, scp = base.read, base.sha, base.dump, base.stamp, base.remote, base.scp
LOCAL = None


def configure(batch):
    global LOCAL
    p.configure(batch)
    LOCAL = BASE/p.RUN
    for module in (base, transport):
        module.p, module.LOCAL, module.MOD, module.DOC, module.AUTH = p, LOCAL, MOD, DOC, AUTH
        module.scope, module.update = scope, update
    transport.PARENT = PARENT


def scope():
    c = read(CONTROL)
    assert c['completed_rounds']==3 and c['automation_status']=='PAUSED' and c['previous_work_paused_by_human']
    assert read(SOURCE_FINAL/'LOCAL_VERIFICATION.json')['passed']
    assert read(TARGET_FINAL/'LOCAL_VERIFICATION.json')['passed']
    if p.BATCH == 'remaining':
        first=BASE/'ffpp_latest_dual_ablation_seed2027_20261105/FINAL_DELIVERY.json'
        delivered=read(first)
        assert delivered['passed'] and delivered['seeds']==[2027]
        assert sha(ROOT/delivered['target_verification'])==delivered['target_verification_sha256']
    return c


def update(phase, proof, **changes):
    c=scope()
    item=c['current_round'].get('latest_dual_ablation', {}).copy()
    batch=item.get('batches', {}).get(p.BATCH, {}).copy()
    batch.update(changes)
    batch.update(run=p.RUN,seeds=list(p.SEEDS),new_arms=list(p.NEW_ARMS),
        retained_update_budget=p.configuration()['retained_updates'])
    item.setdefault('batches', {})[p.BATCH]=batch
    item.update(active_batch=p.BATCH,all_seeds=list(p.ALL_SEEDS),all_new_arms=list(p.NEW_ARMS),
        total_authorized_retained_updates=107850,semantic_unchanged=True,full_reference_reused=True,
        protocol=DOC,old_tasks_and_monitor_paused=True)
    from update_maskfree_control_20261102 import update as atomic
    atomic(sha(CONTROL),dict(human_authorization=AUTH,latest_dual_ablation=item,
        status=batch['status'],source_run=p.RUN,scientific_protocol=DOC,document=DOC,
        source_registration_sha256=batch.get('registration_sha256'),
        registered=batch.get('registered',False),uploaded=batch.get('uploaded',False),started=batch.get('started',False),
        cuda_preflight_passed=batch.get('source_preflight_passed',False),active_seeds=list(p.SEEDS),
        semantic_branch_frozen=True,new_CLIP_training_authorized=False,
        source_only_now=False,seven_domain_test_authorized_now=True,monitor_resume_authorized=False,
        next_action=batch.get('next_action','Only complete five registered latest-model ablations; source archive verification before target; first-seed delivery before remaining seeds')),
        phase,Path(proof).resolve().relative_to(ROOT).as_posix())


def tests():
    q=subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'tools/test_latest_dual_ablation_20261105.py')],
        capture_output=True,text=True,encoding='utf-8')
    assert q.returncode==0,q.stdout+q.stderr
    return dict(passed=True,scope='stdlib AST/protocol/budget/reference only; not CUDA or efficacy',stdout=q.stdout,stderr=q.stderr)


def resources_snapshot():
    return remote(f'''import pathlib,json,subprocess,shutil,datetime
r=pathlib.Path({p.REMOTE!r});cg=pathlib.Path('/sys/fs/cgroup/memory')
out=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),GPU=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True),disk_free=shutil.disk_usage(r).free,memory_cap=int((cg/'memory.limit_in_bytes').read_text()),memory_used=int((cg/'memory.usage_in_bytes').read_text()),new_run_exists=(r/{p.RUN!r}).exists(),prior_processes={{}})
for folder,prefix in [({p.PARENT!r},'SOURCE'),({p.REFERENCE_TARGET!r},'TARGET')]:
 a=json.loads((r/folder/(prefix+'_LAUNCH.json')).read_text());proc=pathlib.Path('/proc')/str(a['pid'])/'cmdline'
 out['prior_processes'][folder+'/'+prefix]=proc.read_bytes().decode().replace(chr(0),' ') if proc.exists() else None
 assert json.loads((r/folder/(prefix+'_COMPLETE.json')).read_text())['passed']
print(json.dumps(out))''')


def prepare():
    scope()
    assert not LOCAL.exists(),'Preserve existing preparation'
    check,snapshot=tests(),resources_snapshot()
    assert not snapshot['new_run_exists'] and not snapshot['GPU'].strip() and not any(snapshot['prior_processes'].values())
    allowance=p.configuration()['total_remaining_output_allowance_bytes'] if p.BATCH=='first' else p.configuration()['output_allowance_bytes']
    assert snapshot['disk_free']>allowance+p.GIB,'Insufficient reserve for all authorized remaining outputs'
    assert snapshot['memory_cap']-snapshot['memory_used']>p.configuration()['train_uint8_bytes']+7*p.GIB
    LOCAL.mkdir()
    dump(LOCAL/'AUTHORIZATION.json',dict(time=stamp(),authorization=AUTH,configuration=p.configuration(),
        resources=snapshot,static_tests=check,protocol_sha256=sha(ROOT/DOC),ledger_sha256=sha(CONTROL.with_name('ROUNDS.jsonl'))))
    update('Latest_dual_ablation_'+p.BATCH+'_preparing',LOCAL/'AUTHORIZATION.json',status='preparing',
        registered=False,uploaded=False,started=False,complete=False,source_preflight_passed=False)
    print(json.dumps(snapshot))


def register():
    scope()
    assert LOCAL.is_dir() and not (LOCAL/'REGISTRATION.json').exists()
    check=tests();old=read(PARENT/'REGISTRATION.json')
    assert old['registration_sha256']==p.PARENT_REG==p.fingerprint({k:v for k,v in old.items() if k!='registration_sha256'})
    for name,digest in old['files'].items():assert sha(PARENT/'bundle'/name)==digest,name
    bundle=LOCAL/'bundle';bundle.mkdir()
    skip={'implementation/plan.py','implementation/model_cuda.py','implementation/patch_config.py','implementation/run_cuda.py',
          'LOCAL_MANAGER.py','LOCAL_VERIFIER.py','SCIENTIFIC_PROTOCOL.md','STATIC_TEST_SOURCE.py','STATIC_TESTS.json','EXTERNAL_LOCK.json','BUNDLE_TRANSFORM_SOURCE.py'}
    def copy(source,name):
        dst=bundle/name;dst.parent.mkdir(parents=True,exist_ok=True);assert not dst.exists();shutil.copyfile(source,dst)
    def generated(name,text):
        dst=bundle/name;dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    for name in old['files']:
        if name not in skip:copy(PARENT/'bundle'/name,name)
    plan=(MOD/'plan.py').read_text(encoding='utf-8')
    assert plan.count("BATCH = 'first'")==1
    generated('implementation/plan.py',plan.replace("BATCH = 'first'",f'BATCH = {p.BATCH!r}'))
    original=PARENT/'bundle/implementation'
    generated('implementation/model_cuda.py',model((original/'model_cuda.py').read_text(encoding='utf-8')))
    generated('implementation/patch_config.py',patch_config((original/'patch_config.py').read_text(encoding='utf-8')))
    copy(original/'patch_config.py','implementation/original_patch_config.py')
    generated('implementation/run_cuda.py',runner((original/'run_cuda.py').read_text(encoding='utf-8')))
    for source,name in ((ROOT/DOC,'SCIENTIFIC_PROTOCOL.md'),(Path(__file__),'LOCAL_MANAGER.py'),
        (ROOT/'tools/build_latest_dual_ablation_20261105.py','BUNDLE_TRANSFORM_SOURCE.py'),
        (ROOT/'tools/test_latest_dual_ablation_20261105.py','STATIC_TEST_SOURCE.py'),
        (ROOT/'tools/verify_latest_dual_ablation_20261105.py','LOCAL_VERIFIER.py'),
        (SOURCE_FINAL/'run/SELECTED_ENDPOINT_LOCK.json','REFERENCE_FULL_ENDPOINTS.json'),
        (SOURCE_FINAL/'run/INITIALIZATION.json','REFERENCE_FULL_INITIALIZATION.json'),
        (SOURCE_FINAL/'LOCAL_VERIFICATION.json','REFERENCE_FULL_SOURCE_VERIFICATION.json'),
        (TARGET_FINAL/'LOCAL_VERIFICATION.json','REFERENCE_FULL_TARGET_VERIFICATION.json'),
        (TARGET_FINAL/'RESULTS.json','REFERENCE_FULL_RESULTS.json')):
        copy(source,name)
    selected=read(SOURCE_FINAL/'run/SELECTED_ENDPOINT_LOCK.json')
    external=read(PARENT/'bundle/EXTERNAL_LOCK.json')
    external[p.REMOTE+'/'+p.PARENT+'/REGISTRATION.json']=sha(PARENT/'REGISTRATION.json')
    external[p.REMOTE+'/'+p.REFERENCE_TARGET+'/REGISTRATION.json']=sha(BASE/p.REFERENCE_TARGET/'REGISTRATION.json')
    for seed in p.SEEDS:
        endpoint=selected[str(seed)]['PatchClassificationContrast']
        sourcepath=SOURCE_FINAL/'run'/Path(endpoint['frames_path']).as_posix().split('/'+p.PARENT+'/',1)[1]
        assert sha(sourcepath)==endpoint['frames_sha256']
        copy(sourcepath,'reference_full_source/'+str(seed)+'.csv')
        external[endpoint['path']]=endpoint['sha256']
    for domain in p.DOMAINS:
        source=TARGET_FINAL/'run/target'/(domain+'_FRAMES.csv')
        assert sha(source)==read(TARGET_FINAL/'run/target'/(domain+'_COMPLETE.json'))['frames_sha256']
        copy(source,'reference_full_target/'+domain+'.csv')
    dump(bundle/'EXTERNAL_LOCK.json',external);dump(bundle/'STATIC_TESTS.json',check)
    lock=dict(configuration=p.configuration(),files={f.relative_to(bundle).as_posix():sha(f) for f in bundle.rglob('*') if f.is_file()},
        entrypoint='implementation/run_cuda.py',ledger_sha256=sha(CONTROL.with_name('ROUNDS.jsonl')))
    lock['registration_sha256']=p.fingerprint(lock)
    dump(bundle/'REGISTRATION.json',lock);dump(LOCAL/'REGISTRATION.json',lock)
    update('Latest_dual_ablation_'+p.BATCH+'_registered',LOCAL/'REGISTRATION.json',status='registered',registered=True,registration_sha256=lock['registration_sha256'])
    print(json.dumps(dict(registration_sha256=lock['registration_sha256'],registered_files=len(lock['files']),retained_updates=p.configuration()['retained_updates'])))


def verify_source_preflight():
    lock=base.verified();n=len(p.SEEDS)*len(p.NEW_ARMS)
    proof=remote(f'''import pathlib,json,subprocess,datetime
r=pathlib.Path({p.REMOTE!r})/{p.RUN!r};read=lambda p:json.loads(p.read_text())
assert not (r/'SOURCE_PREFLIGHT_FAILURE.json').exists() and not (r/'SOURCE_LAUNCH.json').exists()
done=read(r/'SOURCE_PREFLIGHT_COMPLETE.json');launch=read(r/'SOURCE_PREFLIGHT_LAUNCH.json')
assert done['passed'] and done['registration_sha256']=={lock['registration_sha256']!r}
assert done['new_retained_updates']==0 and done['disposable_optimizer_updates']=={n*2}
assert done['preflight_frame_forwards']=={p.configuration()['source_preflight_total_forwards']}
assert done['network_frame_forwards']==dict(historical_full_replay={len(p.SEEDS)*64},latest_full_replay={len(p.SEEDS)*64},synthetic_zero_head={n*2},real_TRAIN_gradient={n*2},disposable_synthetic={n*4})
assert max(done['historical_full_max_errors'].values())<=2e-7 and max(done['latest_full_max_errors'].values())<=2e-7
assert set(done['initialization_sha256'])=={set(map(str,p.SEEDS))!r}
for seed,checks in done['gradient_checks'].items():
 assert set(checks)=={set(p.NEW_ARMS)!r}
 assert len(set(done['initialization_sha256'][seed].values()))==1
 for c in checks.values():assert c['passed'] and not c['semantic_gradient'] and c['invalid_mask_invariant']
proc=pathlib.Path('/proc')/str(launch['pid'])/'cmdline';assert not proc.exists() or not proc.read_bytes()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
print(json.dumps(dict(passed=True,time=datetime.datetime.now(datetime.timezone.utc).isoformat(),process_exited=True,
 registration_sha256=done['registration_sha256'],complete=done,owner=read(r/'SOURCE_PREFLIGHT_CUDA_OWNER.json'))))''')
    dump(LOCAL/'SOURCE_PREFLIGHT_EXIT_VERIFICATION.json',proof)
    update('Latest_dual_ablation_'+p.BATCH+'_preflight_verified',LOCAL/'SOURCE_PREFLIGHT_EXIT_VERIFICATION.json',
        status='source_preflight_verified_exited',source_preflight_passed=True,source_preflight_verified_exited=True,started=False)
    print(json.dumps(dict(passed=True,process_exited=True,latest_full_max_errors=proof['complete']['latest_full_max_errors'],gradient_checks=n)))


def startup():
    lock=base.verified()
    proof=remote(f'''import pathlib,json,subprocess,datetime
r=pathlib.Path({p.REMOTE!r})/{p.RUN!r};read=lambda p:json.loads(p.read_text())
assert not (r/'SOURCE_FAILURE.json').exists()
launch=read(r/'SOURCE_LAUNCH.json');proc=pathlib.Path('/proc')/str(launch['pid'])/'cmdline'
assert proc.exists() and b'--stage\x00source' in proc.read_bytes()
owner=read(r/'SOURCE_CUDA_OWNER.json');assert owner['container_pid']==launch['pid']
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
assert [int(x.strip()) for x in gpu.splitlines() if x.strip().isdigit()]==[owner['nvml_owned_context_pid']]
training=[json.loads(line) for line in (r/'STAGES.jsonl').read_text().splitlines() if json.loads(line)['stage']=='training']
assert training and training[-1]['step']>=100 and training[-1]['arm'] in {list(p.NEW_ARMS)!r}
print(json.dumps(dict(passed=True,time=datetime.datetime.now(datetime.timezone.utc).isoformat(),registration_sha256={lock['registration_sha256']!r},launch=launch,
 latest_training=training[-1],state=read(r/'STATE.json'),owner=owner,target_started=False)))''')
    dump(LOCAL/'STARTUP_VERIFICATION.json',proof)
    update('Latest_dual_ablation_'+p.BATCH+'_training_confirmed',LOCAL/'STARTUP_VERIFICATION.json',status='source_training_confirmed',
        started=True,source_preflight_passed=True,real_training_confirmed=True,first_verified_training=proof['latest_training'])
    print(json.dumps(dict(passed=True,training=proof['latest_training'],proof=str(LOCAL/'STARTUP_VERIFICATION.json'))))


def archive(stage):
    source=inspect.getsource(base.archive)
    assert source.count("tarfile.open(path,'x:gz')")==1
    source=source.replace("tarfile.open(path,'x:gz')","tarfile.open(path,'x:gz',dereference=True)")
    namespace=dict(vars(base));exec(compile(source,'materialized_archive','exec'),namespace)
    namespace['archive'](stage)


def verify_archive(stage,path):
    import verify_latest_dual_ablation_20261105 as verification
    verification.m=sys.modules[__name__]
    result=getattr(verification,'verify_'+stage)(Path(path).resolve())
    print(json.dumps(dict(passed=result['passed'],registration_sha256=result['registration_sha256'])))


configure('first')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('tests','resources','prepare','register','upload','source-preflight',
        'verify-source-preflight','source','startup','status','archive-source','verify-source','upload-source-proof','target-preflight','target','archive-target','verify-target'))
    ap.add_argument('--batch',choices=('first','remaining'),default='first');ap.add_argument('--archive');args=ap.parse_args();configure(args.batch)
    if args.action in ('tests','resources','prepare','register','startup'):
        value=(resources_snapshot if args.action=='resources' else globals()[args.action])()
        if value is not None:print(json.dumps(value))
    elif args.action=='upload':transport.upload()
    elif args.action=='verify-source-preflight':verify_source_preflight()
    elif args.action.startswith('archive-'):archive(args.action.split('-')[1].upper())
    elif args.action in ('verify-source','verify-target'):verify_archive(args.action.split('-')[1],args.archive)
    elif args.action=='upload-source-proof':base.upload_source_proof()
    elif args.action in ('source-preflight','source','target-preflight','target'):
        if args.action=='source':
            proof=read(LOCAL/'SOURCE_PREFLIGHT_EXIT_VERIFICATION.json')
            assert proof['passed'] and proof['process_exited'] and proof['registration_sha256']==base.verified()['registration_sha256']
        base.launch(args.action)
    else:base.status()
