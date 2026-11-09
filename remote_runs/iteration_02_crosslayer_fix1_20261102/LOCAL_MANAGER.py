"""Standard-library builder/transport/status/archive; all model execution remote CUDA."""
import argparse
import ast
import datetime
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import unittest
import manage_v17_patch as m

ROOT, BASE, REMOTE, HOST, SSH = m.ROOT, m.BASE, m.REMOTE, m.HOST, m.SSH
SOURCE = ROOT / 'code/dual_stream_20261026/iteration02_crosslayer_fix1'
sys.path.insert(0, str(SOURCE))
import crosslayer_plan as p

NAME = p.RUN_NAME
ARCHIVE = BASE / NAME
OLD = BASE / 'v22_frame_coverage_fix1_20261101/remote_bundle'
PRIOR_SOURCE = BASE / 'iteration_01_quality_refresh_20261102/final_20261102T034829Z'
PRIOR_TARGET = BASE / 'iteration_01_quality_crossdomain_20261102/final_20261102T042737Z'
CONTROL = BASE / 'iterative_crossdomain_10rounds_20261102/CONTROL.json'
PROTOCOL = ROOT / 'docs/72_iteration02数值布局与CUDA归属独立修复_20261102.md'
DIAGNOSTIC = BASE / 'iteration_02_crosslayer_layout_diagnostic_20261102'
FAILED = BASE / 'iteration_02_crosslayer_20261102'


def tests():
    import test_iteration02_crosslayer_fix1 as module
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    p.require(result.wasSuccessful(), stream.getvalue())
    for file in (*SOURCE.glob('*.py'), Path(__file__), ROOT / 'tools/verify_iteration02_source_fix1.py'):
        ast.parse(file.read_text(encoding='utf-8'), filename=str(file))
    p.require(not any(n in sys.modules for n in ('torch', 'numpy', 'PIL')), 'No local model modules')
    return dict(passed=True, count=result.testsRun, report=stream.getvalue(), local_model_execution=False)


def plan():
    print(json.dumps(dict(tests=tests(), configuration=p.configuration()), ensure_ascii=False))


def register():
    check = tests()
    p.require(not ARCHIVE.exists(), 'Existing registration preserved')
    control = p.read(CONTROL)
    p.require(control['completed_rounds'] == 1 and control['current_round']['iteration'] == 2 and
              not control['current_round']['source_full_training_launched'], 'Wrong bounded-round state')
    diagnosis = p.read(DIAGNOSTIC / 'RESULT.json')
    p.require(p.digest(DIAGNOSTIC / 'RESULT.json') == p.configuration()['implementation_repair']['diagnosis_result_sha256'] and
              diagnosis['corrected_all_probes_exact'] and diagnosis['optimizer_updates'] == 0 and diagnosis['models_cuda_only'], 'Layout diagnosis not independently recorded')
    parent_lock = p.read(FAILED / 'remote_bundle/REGISTRATION.json')
    p.require(parent_lock['registration_sha256'] == p.configuration()['implementation_repair']['failed_parent_registration_sha256'], 'Wrong failed parent')
    scientific = p.configuration().copy()
    scientific.pop('implementation_repair')
    p.require(scientific == parent_lock['configuration'], 'Repair changed scientific protocol')
    for folder in (PRIOR_SOURCE, PRIOR_TARGET):
        p.require(p.read(folder / 'LOCAL_VERIFICATION.json')['passed'], 'Previous round not verified')
        inventory = p.read(folder / 'REMOTE_INVENTORY.json')
        for name in ('FINAL_REPORT.json', 'COMPLETE.json', 'REGISTRATION.json'):
            p.require(p.digest(folder / 'run' / name) == inventory['files'][name]['sha256'] if 'files' in inventory else
                      p.digest(folder / 'run' / name) == inventory[name]['sha256'], 'Previous proof changed')
    p.require(p.digest(PRIOR_TARGET / 'run/FINAL_REPORT.json') == 'cce36c0e8f820718dfc0fe32e39c5c4295b84d903a983a861561e98ccc915064', 'Wrong iteration01 target')
    original_lock = p.read(OLD / 'REGISTRATION.json')
    dependencies = p.read(PRIOR_SOURCE / 'run/EXTERNAL_LOCK.json')
    external = {name: sha for name, sha in dependencies.items()
                if '/guarded_dual_20261025/code/' in name or '/lijiale_correction_blend_20261026/code/' in name
                or '/root/drf_selective_pilot_clip_20261026_01/' in name or '/checkpoints/no_residual/' in name
                or '/v22_frame_coverage_fix1_20261101/VAL_DISK/' in name}
    external['/root/miniconda3/lib/python3.10/site-packages/transformers/models/clip/modeling_clip.py'] = p.CLIP_SOURCE_SHA
    failed_snapshot = p.read(FAILED / 'snapshot_20261102T045022162744Z/REMOTE_STATUS.json')
    p.require(failed_snapshot['PREFLIGHT_process'] is None and not control['current_round']['source_full_training_launched'], 'Failed source was not stopped')
    failure_identity = json.loads(m.remote(f"import pathlib,hashlib,json\nr=pathlib.Path({REMOTE!r})/'iteration_02_crosslayer_20261102'\nassert not (r/'TRAIN_LAUNCH.json').exists()\nread=lambda p:json.loads(p.read_text())\nlaunch=read(r/'PREFLIGHT_LAUNCH.json');proc=pathlib.Path('/proc')/str(launch['pid'])/'cmdline'\nassert not proc.exists() or not proc.read_bytes()\nprint(json.dumps(dict(sha256=hashlib.sha256((r/'PREFLIGHT_FAILURE.json').read_bytes()).hexdigest(),failure=read(r/'PREFLIGHT_FAILURE.json'))))"))
    p.require(failure_identity['failure'] == failed_snapshot['PREFLIGHT_FAILURE.json'], 'Failed parent evidence changed')
    external[REMOTE + '/iteration_02_crosslayer_20261102/PREFLIGHT_FAILURE.json'] = failure_identity['sha256']
    external[REMOTE + '/iteration_02_crosslayer_20261102/REGISTRATION.json'] = p.digest(FAILED / 'remote_bundle/REGISTRATION.json')
    for file in ('V5_CONFIG.json', 'BASELINE_SOURCE_VIDEOS.json', 'SPLIT_PROOF.json', 'inputs/fit.json', 'inputs/source_val.json'):
        p.require(p.digest(OLD / file) == original_lock['bundle_sha256'][file], 'Frozen inherited source inputs changed')
    train, val = [p.read(OLD / 'inputs' / (name + '.json')) for name in ('fit', 'source_val')]
    proof = p.verify_split(train, val)
    p.require(proof['train_families'] == 291 and proof['val_families'] == 70, 'Wrong source family counts')
    for name, prefix in (('iteration_01_quality_refresh_20261102', 'TRAIN'), ('iteration_01_quality_crossdomain_20261102', 'RUN')):
        folder = PRIOR_SOURCE if prefix == 'TRAIN' else PRIOR_TARGET
        for file in ('COMPLETE.json', 'FINAL_REPORT.json'):
            external[f'{REMOTE}/{name}/{file}'] = p.digest(folder / 'run' / file)
    estimate = p.cache_estimate()
    evidence = json.loads(m.remote(f'''
import datetime,hashlib,json,os,pathlib,shutil,subprocess
root=pathlib.Path({REMOTE!r});assert not (root/{NAME!r}).exists(),'Existing iteration preserved'
def read(path):return json.loads(path.read_text())
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
  os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
 return h.hexdigest()
for name,value in {external!r}.items():assert sha(name)==value,name
for name,prefix in (('iteration_01_quality_refresh_20261102','TRAIN'),('iteration_01_quality_crossdomain_20261102','RUN')):
 run=root/name;launch=read(run/(prefix+'_LAUNCH.json'));proc=pathlib.Path('/proc')/str(launch['pid'])/'cmdline'
 assert not proc.exists() or not proc.read_bytes(),'Previous run still alive'
 assert read(run/'COMPLETE.json')['final_report_sha256']==sha(run/'FINAL_REPORT.json')
users=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
assert not any(x.strip().isdigit() for x in users.splitlines()),'External GPU user; manual handling, no queue'
base=pathlib.Path('/sys/fs/cgroup/memory');cap=int((base/'memory.limit_in_bytes').read_text());used=int((base/'memory.usage_in_bytes').read_text())
assert cap-used>={estimate['two_train_views_bytes']!r}+8*1024**3,'Need two TRAIN RAM views,2GiB runtime,6GiB reserve'
free=shutil.disk_usage(root).free
assert free>={estimate['added_val_bytes']!r}+11*1024**3,'Need added VAL,2GiB output,8GiB future estimate,1GiB reserve'
print(json.dumps(dict(time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),gpu=users,disk_free=free,memory_cap=cap,memory_used=used,verified_dependencies=len({external!r}),model_execution=False,source_test_access=False)))
'''))
    bundle = ARCHIVE / 'remote_bundle'
    implementation = bundle / 'implementation'
    implementation.mkdir(parents=True)
    for name in ('frame_plan.py', 'disk_val_cache.py', 'multiscale_cuda.py', 'multiscale_plan.py', 'run_multiscale_cuda.py'):
        file = OLD / 'implementation' / name
        p.require(p.digest(file) == original_lock['bundle_sha256']['implementation/' + name], 'Changed original implementation')
        shutil.copy2(file, implementation / name)
    for folder in ('reference', 'v5'):
        for file in (OLD / 'implementation' / folder).rglob('*'):
            if file.is_file() and '__pycache__' not in file.parts:
                p.require(p.digest(file) == original_lock['bundle_sha256'][file.relative_to(OLD).as_posix()], 'Changed inherited implementation')
        shutil.copytree(OLD / 'implementation' / folder, implementation / folder, ignore=shutil.ignore_patterns('__pycache__'))
    for file in SOURCE.glob('*.py'):
        shutil.copy2(file, implementation / file.name)
    (bundle / 'inputs').mkdir()
    for name in ('fit', 'source_val'):
        shutil.copy2(OLD / 'inputs' / (name + '.json'), bundle / 'inputs' / (name + '.json'))
    for name in ('V5_CONFIG.json', 'BASELINE_SOURCE_VIDEOS.json'):
        shutil.copy2(OLD / name, bundle / name)
    shutil.copy2(PROTOCOL, bundle / 'PREREGISTRATION_CN.md')
    p.require(p.digest(FAILED / 'remote_bundle/PREREGISTRATION_CN.md') == parent_lock['bundle_sha256']['PREREGISTRATION_CN.md'], 'Original scientific preregistration changed')
    shutil.copy2(FAILED / 'remote_bundle/PREREGISTRATION_CN.md', bundle / 'PREREGISTRATION_BASE_CN.md')
    for src, dest in ((Path(__file__), 'LOCAL_MANAGER.py'), (ROOT / 'tools/test_iteration02_crosslayer_fix1.py', 'LOCAL_PROTOCOL_TESTS.py'),
                      (ROOT / 'tools/verify_iteration02_source_fix1.py', 'LOCAL_VERIFIER.py')):
        shutil.copy2(src, bundle / dest)
    m.dump(bundle / 'REPAIR_EVIDENCE.json', dict(failed_parent_registration=parent_lock['registration_sha256'],
           parent_configuration=parent_lock['configuration'], failed_preflight=failure_identity,
           diagnostic=diagnosis, diagnostic_local_sha256=p.digest(DIAGNOSTIC / 'RESULT.json'),
           scientific_protocol_unchanged=True, old_failure_preserved=True, no_old_training_updates=True))
    m.dump(bundle / 'EXTERNAL_LOCK.json', external)
    m.dump(bundle / 'SPLIT_PROOF.json', proof)
    m.dump(bundle / 'PREVIOUS_ROUND_PROOF.json', dict(source=p.read(PRIOR_SOURCE / 'LOCAL_VERIFICATION.json'), target=p.read(PRIOR_TARGET / 'LOCAL_VERIFICATION.json'),
            target_report_sha256=p.digest(PRIOR_TARGET / 'run/FINAL_REPORT.json'), target_refresh_macro_gain_pp=-0.013635656914290806))
    m.dump(ARCHIVE / 'RESOURCE_IDENTITY_CHECK.json', evidence)
    m.dump(ARCHIVE / 'LOCAL_TESTS.json', check)
    files = {file.relative_to(bundle).as_posix(): p.digest(file) for file in sorted(bundle.rglob('*')) if file.is_file()}
    preplan = dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), configuration=p.configuration(), bundle_sha256=files, tests=dict(passed=True, count=check['count']))
    m.dump(ARCHIVE / 'PREPLAN.json', preplan)
    lock = dict(schema='bounded-iteration02-crosslayer-three-arm-v1', **preplan, remote_name=NAME, entrypoint='implementation/run_crosslayer_cuda.py')
    lock['registration_sha256'] = p.fingerprint(lock)
    m.dump(bundle / 'REGISTRATION.json', lock)
    print(json.dumps(dict(registration_sha256=lock['registration_sha256'], tests=check['count'], model_started=False, completed_rounds=1)))


def upload():
    bundle = ARCHIVE / 'remote_bundle'
    lock = p.read(bundle / 'REGISTRATION.json')
    for name, sha in lock['bundle_sha256'].items():
        p.require(p.digest(bundle / name) == sha, 'Frozen file changed')
    print(m.remote(f"import pathlib\nr=pathlib.Path({REMOTE!r})/{NAME!r}\nassert not r.exists(),'Existing remote preserved'\nr.mkdir()\nprint('NEW_ITERATION02_DIRECTORY')"))
    for file in bundle.iterdir():
        subprocess.run(['scp', '-P', '28081', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-r', str(file), f'{HOST}:{REMOTE}/{NAME}/'], check=True)
    result = json.loads(m.remote(f"import pathlib,json,hashlib\nr=pathlib.Path({REMOTE!r})/{NAME!r}\nlock=json.loads((r/'REGISTRATION.json').read_text())\nfor name,sha in lock['bundle_sha256'].items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==sha,name\nprint(json.dumps(dict(passed=True,registration_sha256=lock['registration_sha256'],models_started=False,val_cache_duplicated=False)))"))
    m.dump(ARCHIVE / 'UPLOAD_VERIFICATION.json', result)
    print(json.dumps(result))


def start(preflight=False):
    prefix = 'PREFLIGHT' if preflight else 'TRAIN'
    result = json.loads(m.remote(f'''
import datetime,json,os,pathlib,shutil,subprocess
r=pathlib.Path({REMOTE!r})/{NAME!r};prefix={prefix!r};read=lambda p:json.loads(p.read_text())
for name in (prefix+'_LAUNCH.json',prefix+'_STARTED.json',prefix+'_FAILURE.json',prefix+'.log','COMPLETE.json'):assert not (r/name).exists(),'Existing stage preserved'
lock=read(r/'REGISTRATION.json')
if prefix=='TRAIN':
 assert not (r/'PREFLIGHT_FAILURE.json').exists(),'Preflight failed'
 check=read(r/'PREFLIGHT.json');assert check['passed'] and check['registration_sha256']==lock['registration_sha256']
 proc=pathlib.Path('/proc')/str(read(r/'PREFLIGHT_LAUNCH.json')['pid'])/'cmdline';assert not proc.exists() or not proc.read_bytes(),'Preflight still alive'
users=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
assert not any(x.strip().isdigit() for x in users.splitlines()),'External GPU user; manual handling, no queue'
assert shutil.disk_usage(r).free>={p.cache_estimate()['added_val_bytes']!r}+3*1024**3,'Need early VAL,output and1GiB reserve'
cmd=['/root/miniconda3/bin/python','-u',str(r/lock['entrypoint']),'--run',str(r)]
if prefix=='PREFLIGHT':cmd.append('--preflight-only')
env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
with (r/(prefix+'.log')).open('xb') as dst:proc=subprocess.Popen(cmd,cwd=r,env=env,stdout=dst,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
out=dict(time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),pid=proc.pid,command=cmd,registration_sha256=lock['registration_sha256'],source_test_access=False)
with (r/(prefix+'_LAUNCH.json')).open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out))
'''))
    m.dump(ARCHIVE / (prefix + '_LAUNCH.json'), result)
    print(json.dumps(result))


def status():
    result = json.loads(m.remote(f'''
import datetime,json,pathlib,shutil,subprocess
r=pathlib.Path({REMOTE!r})/{NAME!r};b=pathlib.Path('/sys/fs/cgroup/memory')
out=dict(time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),exists=r.exists(),gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv'],text=True),disk_free=shutil.disk_usage(r.parent).free,memory={{k:(b/k).read_text().strip() for k in ('memory.limit_in_bytes','memory.usage_in_bytes')}})
for name in ('STATE.json','PREFLIGHT.json','PREFLIGHT_FAILURE.json','TRAIN_FAILURE.json','COMPLETE.json','FINAL_REPORT.json'):
 if (r/name).exists():out[name]=json.loads((r/name).read_text())
for prefix in ('PREFLIGHT','TRAIN'):
 if (r/(prefix+'_LAUNCH.json')).exists():
  launch=json.loads((r/(prefix+'_LAUNCH.json')).read_text());proc=pathlib.Path('/proc')/str(launch['pid'])/'cmdline';out[prefix+'_launch']=launch;out[prefix+'_process']=proc.read_bytes().decode().replace(chr(0),' ') if proc.exists() else None
 if (r/(prefix+'.log')).exists():out[prefix+'_tail']=(r/(prefix+'.log')).read_text(errors='replace').splitlines()[-8:]
print(json.dumps(out))
'''))
    folder = ARCHIVE / ('snapshot_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    folder.mkdir()
    m.dump(folder / 'REMOTE_STATUS.json', result)
    print(json.dumps(result))
    print(folder)


def archive():
    p.require(not any(ARCHIVE.glob('final_*')), 'Existing archive preserved; inspect/resume it, do not repeat')
    inventory = json.loads(m.remote(f'''
import hashlib,json,os,pathlib
r=pathlib.Path({REMOTE!r})/{NAME!r};read=lambda p:json.loads(p.read_text())
assert (r/'COMPLETE.json').exists() and not (r/'TRAIN_FAILURE.json').exists(),'Source incomplete'
proc=pathlib.Path('/proc')/str(read(r/'TRAIN_LAUNCH.json')['pid'])/'cmdline';assert not proc.exists() or not proc.read_bytes(),'Source alive'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
  os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
 return h.hexdigest()
assert read(r/'COMPLETE.json')['final_report_sha256']==sha(r/'FINAL_REPORT.json')
for name,value in read(r/'EXTERNAL_LOCK.json').items():assert sha(pathlib.Path(name))==value,name
cache=r/'VAL_EARLY/patches.bin';layout=read(cache.parent/'LAYOUT.json');assert sha(cache)==layout['sha256'] and cache.stat().st_size==layout['bytes']
files={{f.relative_to(r).as_posix():dict(sha256=sha(f),bytes=f.stat().st_size) for f in r.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f!=cache}}
print(json.dumps(dict(files=files,excluded_reconstructible_cache=dict(path=str(cache),sha256=layout['sha256'],bytes=layout['bytes'],remote_preserved=True),all_external_dependencies_verified=True)))
'''))
    folder = ARCHIVE / ('final_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    folder.mkdir()
    m.dump(folder / 'REMOTE_INVENTORY.json', inventory)
    with (folder / 'run.tar.gz').open('xb') as dst:
        subprocess.run(SSH + [f'tar -C {REMOTE} --exclude=__pycache__ --exclude={NAME}/VAL_EARLY/patches.bin -czf - {NAME}'], stdout=dst, check=True)
    seen = set()
    with tarfile.open(folder / 'run.tar.gz') as tar:
        for member in tar:
            if member.isdir():
                continue
            parts = member.name.split('/')
            name = '/'.join(parts[1:])
            p.require(member.isfile() and parts[0] == NAME and name in inventory['files'] and name not in seen and '..' not in parts and ':' not in member.name and '\\' not in member.name, 'Unsafe/extra archive member')
            seen.add(name)
            out = folder / 'run' / name
            out.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, out.open('xb') as dst:
                shutil.copyfileobj(src, dst)
    p.require(seen == set(inventory['files']), 'Incomplete archive')
    for name, value in inventory['files'].items():
        file = folder / 'run' / name
        p.require(file.stat().st_size == value['bytes'] and p.digest(file) == value['sha256'], 'Archive file SHA mismatch')
    from verify_iteration02_source_fix1 import verify
    proof = verify(folder / 'run')
    m.dump(folder / 'LOCAL_VERIFICATION.json', proof)
    print(json.dumps(proof))
    print(folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('tests', 'plan', 'register', 'upload', 'preflight', 'launch', 'status', 'archive'))
    action = parser.parse_args().action
    if action in ('preflight', 'launch'):
        start(action == 'preflight')
    elif action == 'tests':
        print(json.dumps(tests()))
    else:
        globals()[action]()
