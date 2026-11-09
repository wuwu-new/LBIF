"""Explicit full patch variant; no monitor or automatic experiment expansion."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import manage_paper_components_remaining_20261104 as transport
from build_patch_region_full_20261105 import runner, verifier

base = transport.base
ROOT, BASE, CONTROL, HOST = base.ROOT, base.BASE, base.CONTROL, base.HOST
MOD = ROOT/'code/dual_stream_20261026/patch_region_full_20261105'
spec = importlib.util.spec_from_file_location('patch_full_manager_plan', MOD/'plan.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
LOCAL = BASE/p.RUN
PARENT = BASE/p.PARENT
IMPLEMENTATION = BASE/p.IMPLEMENTATION
DOC = 'docs/125_真实分块学习完整版本三种子全量协议_20261105.md'
AUTH = '用户2026-11-05授权启动全量，最新明确先跑最完整版本，并在参考包三处审查说明后再次要求启动。仅PatchClassificationContrast×2027/3407/4099各10epoch完整源VAL选模及七域；分类单独版暂缓，原E/语义只读复用，不修改语义，不恢复旧任务或监控，不自动扩预算。'
read, sha, dump, stamp, remote, scp = base.read, base.sha, base.dump, base.stamp, base.remote, base.scp


def scope():
    c = read(CONTROL)
    assert c['completed_rounds']==3 and c['automation_status']=='PAUSED' and c['previous_work_paused_by_human']
    assert read(IMPLEMENTATION/'CUDA_EXIT_VERIFICATION.json')['passed']
    assert read(IMPLEMENTATION/'REGISTRATION.json')['registration_sha256']==p.IMPLEMENTATION_REG
    return c


def update(phase, proof, **changes):
    c = scope()
    item = c['current_round'].get('patch_region_full', {}).copy()
    item.update(changes)
    item.update(run=p.RUN, seeds=list(p.SEEDS), new_arms=list(p.NEW_ARMS),
        retained_update_budget=21570, classification_only_deferred=True,
        old_bundles_unchanged=True, semantic_frozen=True)
    from update_maskfree_control_20261102 import update as atomic
    atomic(sha(CONTROL), dict(human_authorization=AUTH, patch_region_full=item,
        status=item['status'], document=DOC, scientific_protocol=DOC, source_run=p.RUN,
        source_registration_sha256=item.get('registration_sha256'),
        registered=item.get('registered', False), uploaded=item.get('uploaded',False),
        started=item.get('started', False), cuda_preflight_passed=item.get('source_preflight_passed',False),
        semantic_branch_frozen=True, new_CLIP_training_authorized=False,
        seven_domain_test_authorized_now=True, source_only_now=False,
        active_seeds=list(p.SEEDS), monitor_resume_authorized=False,
        next_action=item.get('next_action','Only finish registered full patch variant; source verified archive before target; no monitor or other experiments')),
        phase, Path(proof).resolve().relative_to(ROOT).as_posix())


for module in (base, transport):
    module.p, module.LOCAL, module.MOD, module.DOC, module.AUTH = p, LOCAL, MOD, DOC, AUTH
    module.scope, module.update = scope, update
transport.PARENT = PARENT
verified = base.verified


def tests():
    q = subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'tools/test_patch_region_full_20261105.py')],
        capture_output=True,text=True,encoding='utf-8')
    assert q.returncode==0, q.stdout+q.stderr
    return dict(passed=True,scope='stdlib protocol/budget/AST/reference checks only; not CUDA or efficacy',stdout=q.stdout,stderr=q.stderr)


def resources_snapshot():
    return remote(f'''import pathlib,json,subprocess,shutil,datetime
r=pathlib.Path({p.REMOTE!r});cg=pathlib.Path('/sys/fs/cgroup/memory')
out=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),GPU=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True),disk_free=shutil.disk_usage(r).free,memory_cap=int((cg/'memory.limit_in_bytes').read_text()),memory_used=int((cg/'memory.usage_in_bytes').read_text()),new_run_exists=(r/{p.RUN!r}).exists(),prior_processes={{}})
for folder,prefix in [({p.PARENT!r},'TARGET')]:
 launch=json.loads((r/folder/(prefix+'_LAUNCH.json')).read_text());proc=pathlib.Path('/proc')/str(launch['pid'])/'cmdline'
 out['prior_processes'][folder+'/'+prefix]=proc.read_bytes().decode().replace(chr(0),' ') if proc.exists() else None
 assert json.loads((r/folder/(prefix+'_COMPLETE.json')).read_text())['passed']
out['implementation_complete']=json.loads((r/{p.IMPLEMENTATION!r}/'CUDA_COMPLETE.json').read_text())['passed']
print(json.dumps(out))''')


def prepare():
    scope()
    assert not LOCAL.exists(),'Preserve all preparation evidence'
    check, resources = tests(), resources_snapshot()
    assert not resources['new_run_exists'] and not resources['GPU'].strip()
    assert not any(resources['prior_processes'].values()) and resources['implementation_complete']
    assert resources['disk_free']>3*p.GIB
    assert resources['memory_cap']-resources['memory_used']>p.configuration()['train_uint8_bytes']+7*p.GIB
    LOCAL.mkdir()
    dump(LOCAL/'AUTHORIZATION.json',dict(time=stamp(),authorization=AUTH,configuration=p.configuration(),
        resources=resources,static_tests=check,ledger_sha256=sha(CONTROL.with_name('ROUNDS.jsonl'))))
    update('Patch_full_preparing',LOCAL/'AUTHORIZATION.json',status='preparing',registered=False,
        uploaded=False,started=False,complete=False,source_preflight_passed=False)
    print(json.dumps(resources))


def register():
    scope()
    assert LOCAL.is_dir() and not (LOCAL/'REGISTRATION.json').exists()
    check = tests()
    old, implementation = read(PARENT/'REGISTRATION.json'), read(IMPLEMENTATION/'REGISTRATION.json')
    for folder, lock, digest in ((PARENT,old,p.PARENT_REG),(IMPLEMENTATION,implementation,p.IMPLEMENTATION_REG)):
        assert lock['registration_sha256']==digest==p.fingerprint({k:v for k,v in lock.items() if k!='registration_sha256'})
        for name,value in lock['files'].items():assert sha(folder/'bundle'/name)==value, name
    bundle = LOCAL/'bundle'
    bundle.mkdir()
    skip = {'implementation/plan.py','implementation/model_cuda.py','implementation/run_cuda.py',
        'LOCAL_MANAGER.py','LOCAL_VERIFIER.py','SCIENTIFIC_PROTOCOL.md','STATIC_TEST_SOURCE.py',
        'STATIC_TESTS.json','EXTERNAL_LOCK.json','CONTINUATION.json','BUNDLE_TRANSFORM_SOURCE.py'}
    for name in old['files']:
        if name not in skip:
            dst=bundle/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PARENT/'bundle'/name,dst)
    def copy(src,name):
        dst=bundle/name;dst.parent.mkdir(parents=True,exist_ok=True);assert not dst.exists();shutil.copyfile(src,dst)
    def generated(name,content):
        dst=bundle/name;dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open('x',encoding='utf-8',newline='\n') as stream:stream.write(content)
    copy(MOD/'plan.py','implementation/plan.py')
    for name in ('model_cuda.py','patch_config.py'):
        copy(IMPLEMENTATION/'bundle/implementation'/name,'implementation/'+name)
    generated('implementation/run_cuda.py',runner((PARENT/'bundle/implementation/run_cuda.py').read_text(encoding='utf-8')))
    generated('LOCAL_VERIFIER.py',verifier((ROOT/'tools/verify_paper_components_20261104.py').read_text(encoding='utf-8')))
    for src,name in ((ROOT/DOC,'SCIENTIFIC_PROTOCOL.md'),(Path(__file__),'LOCAL_MANAGER.py'),
        (ROOT/'tools/build_patch_region_full_20261105.py','BUNDLE_TRANSFORM_SOURCE.py'),
        (ROOT/'tools/test_patch_region_full_20261105.py','STATIC_TEST_SOURCE.py'),
        (ROOT/'docs/126_ForensicsAdapter三处实现审查与本轮取舍_20261105.md','REFERENCE_COMPONENT_REVIEW.md'),
        (IMPLEMENTATION/'CUDA_EXIT_VERIFICATION.json','IMPLEMENTATION_CUDA_VERIFICATION.json'),
        (IMPLEMENTATION/'REGISTRATION.json','IMPLEMENTATION_REGISTRATION.json')):
        copy(src,name)
    external = read(PARENT/'bundle/EXTERNAL_LOCK.json')
    external[p.REMOTE+'/'+p.PARENT+'/REGISTRATION.json']=sha(PARENT/'REGISTRATION.json')
    external[p.REMOTE+'/'+p.IMPLEMENTATION+'/REGISTRATION.json']=sha(IMPLEMENTATION/'REGISTRATION.json')
    for name in ('model_cuda.py','patch_config.py'):
        external[p.REMOTE+'/'+p.IMPLEMENTATION+'/implementation/'+name]=sha(IMPLEMENTATION/'bundle/implementation'/name)
    dump(bundle/'EXTERNAL_LOCK.json',external)
    dump(bundle/'STATIC_TESTS.json',check)
    dump(bundle/'PATCH_FULL_SCOPE.json',dict(only_new_arm=list(p.NEW_ARMS),seeds=list(p.SEEDS),
        old_E_reused_not_renamed=True,classification_only_deferred=True,
        no_semantic_model_change=True,implementation_registration_sha256=p.IMPLEMENTATION_REG))
    lock=dict(configuration=p.configuration(),files={f.relative_to(bundle).as_posix():sha(f) for f in bundle.rglob('*') if f.is_file()},
        entrypoint='implementation/run_cuda.py',ledger_sha256=sha(CONTROL.with_name('ROUNDS.jsonl')))
    lock['registration_sha256']=p.fingerprint(lock)
    dump(bundle/'REGISTRATION.json',lock);dump(LOCAL/'REGISTRATION.json',lock)
    update('Patch_full_registered',LOCAL/'REGISTRATION.json',status='registered',registered=True,
        registration_sha256=lock['registration_sha256'])
    print(json.dumps(dict(registration_sha256=lock['registration_sha256'],files=len(lock['files']))))


def verify_source_preflight():
    lock=verified()
    proof=remote(f'''import pathlib,json,subprocess,datetime
r=pathlib.Path({p.REMOTE!r})/{p.RUN!r};read=lambda p:json.loads(p.read_text())
assert not (r/'SOURCE_PREFLIGHT_FAILURE.json').exists() and not (r/'SOURCE_LAUNCH.json').exists()
done=read(r/'SOURCE_PREFLIGHT_COMPLETE.json');launch=read(r/'SOURCE_PREFLIGHT_LAUNCH.json')
assert done['passed'] and done['registration_sha256']=={lock['registration_sha256']!r}
assert done['preflight_frame_forwards']==216 and done['disposable_optimizer_updates']==6 and done['new_retained_updates']==0
assert done['network_frame_forwards']==dict(historical_full_replay=192,synthetic_zero_head=6,real_TRAIN_gradient=6,disposable_synthetic=12)
assert set(done['initialization_sha256'])=={{'2027','3407','4099'}}
assert max(done['historical_full_max_errors'].values())<=2e-7
for checks in done['gradient_checks'].values():
 assert set(checks)=={{'PatchClassificationContrast'}}
 for c in checks.values():assert c['passed'] and c['real_TRAIN_patch_encoder_gradient'] and c['real_TRAIN_contrast_encoder_gradient']
proc=pathlib.Path('/proc')/str(launch['pid'])/'cmdline';assert not proc.exists() or not proc.read_bytes()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
print(json.dumps(dict(passed=True,time=datetime.datetime.now(datetime.timezone.utc).isoformat(),process_exited=True,
 registration_sha256=done['registration_sha256'],complete=done,owner=read(r/'SOURCE_PREFLIGHT_CUDA_OWNER.json'))))''')
    dump(LOCAL/'SOURCE_PREFLIGHT_EXIT_VERIFICATION.json',proof)
    update('Patch_full_source_preflight_verified',LOCAL/'SOURCE_PREFLIGHT_EXIT_VERIFICATION.json',
        status='source_preflight_verified_exited',source_preflight_passed=True,source_preflight_verified_exited=True,started=False)
    print(json.dumps(dict(passed=True,process_exited=True,checks=proof['complete']['gradient_checks'])))


def archive(stage):
    # Administrative packaging fix: materialize hardlinks in the archive.
    import inspect
    source=inspect.getsource(base.archive)
    before="tarfile.open(path,'x:gz')"
    assert source.count(before)==1
    source=source.replace(before,"tarfile.open(path,'x:gz',dereference=True)")
    namespace=dict(vars(base));exec(compile(source,'materialized_archive','exec'),namespace)
    namespace['archive'](stage)


def verify_archive(stage,path):
    location=LOCAL/'bundle/LOCAL_VERIFIER.py'
    spec=importlib.util.spec_from_file_location('patch_full_frozen_verifier',location)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    proof=(module.verify_source if stage=='source' else module.verify_target)(Path(path).resolve())
    print(json.dumps(dict(passed=proof['passed'],registration_sha256=proof['registration_sha256'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('tests','resources','prepare','register','upload','source-preflight',
        'verify-source-preflight','source','status','archive-source','verify-source','upload-source-proof',
        'target-preflight','target','archive-target','verify-target'))
    parser.add_argument('--archive');args=parser.parse_args()
    if args.action in ('tests','resources','prepare','register'):
        result=(resources_snapshot if args.action=='resources' else globals()[args.action])()
        if result is not None:print(json.dumps(result))
    elif args.action=='upload':transport.upload()
    elif args.action=='verify-source-preflight':verify_source_preflight()
    elif args.action.startswith('archive-'):archive(args.action.split('-')[1].upper())
    elif args.action in ('verify-source','verify-target'):verify_archive(args.action.split('-')[1],args.archive)
    elif args.action=='upload-source-proof':base.upload_source_proof()
    elif args.action in ('source-preflight','source','target-preflight','target'):
        if args.action=='source':
            proof=read(LOCAL/'SOURCE_PREFLIGHT_EXIT_VERIFICATION.json')
            assert proof['passed'] and proof['process_exited'] and proof['registration_sha256']==verified()['registration_sha256']
        base.launch(args.action)
    else:base.status()
