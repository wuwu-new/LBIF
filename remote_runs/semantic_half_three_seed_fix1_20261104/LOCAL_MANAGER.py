"""Immutable selected-model manager. No old round or automation launch calls."""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'experiments/guarded_transfer_20261026'
MOD=ROOT/'code/dual_stream_20261026/semantic_half_three_seed_fix1_20261104'
sys.path.insert(0,str(MOD));import confirmation_plan as q
LOCAL=BASE/'best_four_three_seed_confirmation_fix1_20261104'
CONTROL=BASE/'iterative_crossdomain_10rounds_20261102/CONTROL.json'
OLD=BASE/q.PARENT/'remote_bundle';OLDFINAL=BASE/q.PARENT/'final_20261102T062814Z'
HISTORY=BASE/q.HISTORY/'final_20261102T113604Z'
HOST='root@region-46.seetacloud.com'
from manage_ffpp_boundary_readout_target_20261103 import remote,scp
from prepare_best_four_three_seed_20261104 import AUTH


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def dump(path,x):
    with path.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def update(changes,phase,proof):
    c=read(CONTROL);assert c['completed_rounds']==3 and c['automation_status']=='PAUSED'
    stage=c['current_round']['best_four_three_seed_confirmation'].copy();stage.update(changes)
    from update_maskfree_control_20261102 import update as atomic
    atomic(sha(CONTROL),dict(human_authorization=AUTH,best_four_three_seed_confirmation=stage,
        next_action=stage.get('next_action','Only complete authorized selected SemanticHalf three-seed full experiment; old tasks remain paused')),
        phase,proof.relative_to(ROOT).as_posix())
def tests():
    r=subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'tools/test_semantic_half_three_seed_fix1_20261104.py')],capture_output=True,text=True,encoding='utf-8')
    assert r.returncode==0,r.stdout+r.stderr
    return dict(passed=True,scope='stdlib synthetic and static only, not CUDA',stdout=r.stdout,stderr=r.stderr)


def mapped_old_source():
    prior=BASE/'ffpp_boundary_readout_paired_20261103/final_20261103T084508316349Z/run'
    inventory=read(prior/'REMOTE_INVENTORY.json')
    for name in ('SEMANTIC_VAL.csv','inputs/source_val.json'):assert sha(prior/name)==inventory[name]['sha256']
    native=read(prior/'inputs/source_val.json');original=read(OLD/'inputs/source_val.json')
    with (prior/'SEMANTIC_VAL.csv').open(newline='') as f:
        reader=csv.DictReader(f);assert reader.fieldnames==['index','logit'];raw=list(reader)
    assert len(native)==len(original)==len(raw)==22080
    bypath={r['prior_aligned_image_path']:(i,r) for i,r in enumerate(native)};assert len(bypath)==22080
    values=[];mapping=[]
    for index,row in enumerate(original):
        oldindex,record=bypath[row['image_path']]
        assert record['prior_aligned_sha256']==row['sha256']
        assert all(record[k]==row[k] for k in ('video_id','family_id','label','frame_index','source_frame_number'))
        assert int(raw[oldindex]['index'])==oldindex;values.append(float(raw[oldindex]['logit']))
        mapping.append(dict(original_index=index,historical_csv_index=oldindex,image_path=row['image_path'],sha256=row['sha256'],video_id=row['video_id'],family_id=row['family_id'],label=row['label'],frame_index=row['frame_index']))
    assert len({r['historical_csv_index'] for r in mapping})==22080
    return values,mapping,prior


def register():
    assert read(LOCAL/'SELECTION.json')['winner']=='SemanticHalf' and not (LOCAL/'REGISTRATION.json').exists()
    check=tests();bundle=LOCAL/'bundle';bundle.mkdir();oldlock=read(OLD/'REGISTRATION.json');evidence={}
    assert oldlock['registration_sha256']==q.PARENT_REG and read(OLDFINAL/'LOCAL_VERIFICATION.json')['passed']
    for name,value in oldlock['bundle_sha256'].items():
        if name.startswith('implementation/') or name in ('inputs/fit.json','inputs/source_val.json','V5_CONFIG.json','SPLIT_PROOF.json'):
            assert sha(OLD/name)==value;dst=bundle/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OLD/name,dst)
    guard=OLD/'implementation/reference/gpu_identity_guard.py';shutil.copyfile(guard,bundle/'implementation/gpu_identity_guard.py')
    for path in MOD.glob('*.py'):shutil.copyfile(path,bundle/'implementation'/path.name)
    src=BASE/'ffpp_boundary_readout_paired_20261103/final_20261103T084508316349Z/run/SEMANTIC_VAL.csv'
    source_inventory=read(src.parents[0]/'REMOTE_INVENTORY.json');assert sha(src)==source_inventory['SEMANTIC_VAL.csv']['sha256']
    values,mapping,prior_source=mapped_old_source()
    shutil.copyfile(src,bundle/'OLD_SOURCE_RAW_ORDER.csv');dump(bundle/'OLD_SOURCE_MAPPING.json',mapping)
    with (bundle/'OLD_SOURCE_FRAMES.csv').open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(['index','logit']);writer.writerows(enumerate(values))
    evidence[(prior_source/'inputs/source_val.json').relative_to(ROOT).as_posix()]=sha(prior_source/'inputs/source_val.json')
    evidence[src.relative_to(ROOT).as_posix()]=sha(src)
    dump(bundle/'SOURCE_PROBES.json',[dict(start=i,logits=values[i:i+16]) for i in (0,5504,11040,16560)])
    prior=read(OLDFINAL/'run/arms/CrossLayerNoImage/epoch_10.json');init=read(OLDFINAL/'run/PREFLIGHT.json')['model_initialization_sha256']['CrossLayerNoImage']
    dump(bundle/'OLD2027.json',dict(path=q.REMOTE+'/'+q.PARENT+'/arms/CrossLayerNoImage/epoch_10.pt',
        checkpoint_sha256=prior['checkpoint_sha256'],model_tensor_sha256=prior['model_tensor_sha256'],initialization_sha256=init,
        seed=2027,epoch=10,registration_sha256=q.PARENT_REG,updates=7190))
    ext=read(LOCAL/'EXTERNAL_LOCK_DRAFT.json');histinv=read(HISTORY/'REMOTE_INVENTORY.json')['files'];references={}
    source_ref=read(HISTORY/'run/TARGET_REFERENCE.json');manifests=BASE/'v17_patch_forensics_eval_fix4_20261101/final_20261101T044650Z/target_manifests'
    (bundle/'manifests').mkdir()
    for domain in q.DOMAINS:
        path=manifests/(domain+'.json');assert sha(path)==source_ref[domain]['manifest_sha256'];shutil.copyfile(path,bundle/'manifests'/path.name)
        frame=HISTORY/'run/evaluation'/(domain+'_FRAMES.jsonl');assert sha(frame)==histinv['evaluation/'+frame.name]['sha256']
        ext[q.REMOTE+'/'+q.HISTORY+'/evaluation/'+frame.name]=sha(frame);references[domain]=dict(baseline_frames=q.REMOTE+'/'+q.HISTORY+'/evaluation/'+frame.name)
        evidence[frame.relative_to(ROOT).as_posix()]=sha(frame)
        for parent in (HISTORY/'run',BASE/'v22_crossdomain_fix2_pathfix1_20261102/final_20261102T024341Z/run'):
            history_file=parent/'evaluation'/(domain+'_VIDEOS.json')
            evidence[history_file.relative_to(ROOT).as_posix()]=sha(history_file)
    px=HISTORY/'run/TARGET_PIXEL_SHA256.json';assert sha(px)==histinv['TARGET_PIXEL_SHA256.json']['sha256'];ext[q.REMOTE+'/'+q.HISTORY+'/'+px.name]=sha(px)
    dump(bundle/'TARGET_REFERENCE.json',references);dump(bundle/'EXTERNAL_LOCK.json',ext)
    for path in (LOCAL/'AUTHORIZATION.json',LOCAL/'SELECTION.json',LOCAL/'RESOURCE_AND_INPUT_AUDIT.json',OLDFINAL/'LOCAL_VERIFICATION.json',
        BASE/'semantic_half_crossdomain_20261103/RESULTS.json',BASE/'semantic_half_crossdomain_20261103/LOCAL_VERIFICATION.json',
        BASE/'ffpp_boundary_readout_fixed_fusion_20261103/RESULTS.json'):
        evidence[path.relative_to(ROOT).as_posix()]=sha(path)
    for path,name in ((LOCAL/'AUTHORIZATION.json','AUTHORIZATION.json'),(LOCAL/'SELECTION.json','SELECTION.json'),
        (ROOT/'docs/109_SemanticHalf三种子CUDA加载独立修复_20261104.md','SCIENTIFIC_PROTOCOL.md'),
        (ROOT/'docs/108_SemanticHalf三种子完整执行冻结协议_20261104.md','PARENT_SCIENTIFIC_PROTOCOL.md'),
        (Path(__file__),'LOCAL_MANAGER.py'),(ROOT/'tools/analyze_semantic_half_three_seed_fix1_20261104.py','LOCAL_ANALYZER.py'),
        (ROOT/'tools/verify_semantic_half_three_seed_fix1_20261104.py','LOCAL_VERIFIER.py'),
        (ROOT/'tools/test_semantic_half_three_seed_fix1_20261104.py','STATIC_TEST_SOURCE.py'),
        (LOCAL/'FAILED_PARENT_EVIDENCE.json','FAILED_PARENT_EVIDENCE.json')):shutil.copyfile(path,bundle/name)
    dump(bundle/'STATIC_TESTS.json',check)
    lock=dict(time=stamp(),configuration=q.configuration(),files={p.relative_to(bundle).as_posix():sha(p) for p in bundle.rglob('*') if p.is_file()},
        local_evidence_sha256=evidence,ledger_sha256=sha(CONTROL.with_name('ROUNDS.jsonl')))
    lock['registration_sha256']=q.fingerprint(lock);dump(bundle/'REGISTRATION.json',lock);dump(LOCAL/'REGISTRATION.json',lock)
    update(dict(registered=True,uploaded=False,started=False,CUDA_preflight_passed=False,
        directory=LOCAL.relative_to(ROOT).as_posix(),failed_parent='semantic_half_three_seed_20261104',
        failed_parent_retained_updates=0,failed_parent_disposable_updates=0,
        registration_sha256=lock['registration_sha256'],status='registered_not_uploaded'),
        'SemanticHalf_three_seed_registered',LOCAL/'REGISTRATION.json');print(json.dumps(dict(registration_sha256=lock['registration_sha256'],files=len(lock['files']))))


def verified():
    lock=read(LOCAL/'REGISTRATION.json');assert lock['configuration']==q.configuration()
    assert lock['registration_sha256']==q.fingerprint({k:v for k,v in lock.items() if k!='registration_sha256'})
    for name,value in lock['files'].items():assert sha(LOCAL/'bundle'/name)==value
    for name,value in lock['local_evidence_sha256'].items():assert sha(ROOT/name)==value
    assert sha(CONTROL.with_name('ROUNDS.jsonl'))==lock['ledger_sha256'];return lock
def upload():
    lock=verified();remote(f"import pathlib,json;p=pathlib.Path({q.REMOTE!r})/{q.RUN!r};assert not p.exists();print(json.dumps(dict(absent=True)))")
    scp(LOCAL/'bundle',HOST+':'+q.REMOTE+'/'+q.RUN)
    result=remote(f"import pathlib,json,hashlib;p=pathlib.Path({q.REMOTE!r})/{q.RUN!r};l=json.loads((p/'REGISTRATION.json').read_text());assert l['registration_sha256']=={lock['registration_sha256']!r};assert all(hashlib.sha256((p/k).read_bytes()).hexdigest()==v for k,v in l['files'].items());print(json.dumps(dict(passed=True,registration_sha256=l['registration_sha256'])))")
    dump(LOCAL/'UPLOAD_VERIFICATION.json',result);update(dict(uploaded=True,status='uploaded_not_preflighted'),'SemanticHalf_three_seed_uploaded',LOCAL/'UPLOAD_VERIFICATION.json');print(json.dumps(result))


def launch(prefix):
    lock=verified();assert read(LOCAL/'UPLOAD_VERIFICATION.json')['passed']
    result=remote(f'''
import datetime,json,os,pathlib,subprocess,shutil
r=pathlib.Path({q.REMOTE!r})/{q.RUN!r};prefix={prefix!r};read=lambda p:json.loads(p.read_text())
for n in (prefix+'_LAUNCH.json',prefix+'_STARTED.json',prefix+'_FAILURE.json',prefix+'.log'):assert not (r/n).exists(),'Preserve '+n
assert read(r/'REGISTRATION.json')['registration_sha256']=={lock['registration_sha256']!r}
def exited(name):
 proc=pathlib.Path('/proc')/str(read(r/(name+'_LAUNCH.json'))['pid'])/'cmdline';assert not proc.exists() or not proc.read_bytes(),name+' alive'
if prefix=='SOURCE_TRAIN':assert read(r/'SOURCE_PREFLIGHT.json')['passed'];exited('SOURCE_PREFLIGHT')
if prefix.startswith('TARGET'):
 assert read(r/'SOURCE_COMPLETE.json')['registration_sha256']=={lock['registration_sha256']!r}
 assert read(r/'SOURCE_LOCAL_VERIFICATION.json')['passed'];exited('SOURCE_TRAIN')
if prefix=='TARGET_RUN':assert read(r/'TARGET_PREFLIGHT.json')['passed'];exited('TARGET_PREFLIGHT')
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
assert not any(x.strip().isdigit() for x in gpu.splitlines()),'Existing GPU user, no queue'
cg=pathlib.Path('/sys/fs/cgroup/memory');assert int((cg/'memory.limit_in_bytes').read_text())-int((cg/'memory.usage_in_bytes').read_text())>8*1024**3
assert shutil.disk_usage(r).free>3*1024**3
entry='run_source_cuda.py' if prefix.startswith('SOURCE') else 'run_target_cuda.py'
cmd=['/root/miniconda3/bin/python','-u',str(r/'implementation'/entry),'--run',str(r)]
if prefix.endswith('PREFLIGHT'):cmd.append('--preflight-only')
env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
with (r/(prefix+'.log')).open('xb') as f:proc=subprocess.Popen(cmd,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
out=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=proc.pid,command=cmd,registration_sha256={lock['registration_sha256']!r})
with (r/(prefix+'_LAUNCH.json')).open('x') as f:json.dump(out,f)
print(json.dumps(out))
''')
    path=LOCAL/(prefix+'_LAUNCH.json');dump(path,result);update(dict(status=prefix.lower()+'_launched',**{prefix.lower()+'_pid':result['pid']},started=prefix=='SOURCE_TRAIN'),
        'SemanticHalf_three_seed_'+prefix.lower()+'_launched',path);print(json.dumps(result))


def status():
    result=remote(f'''
import datetime,json,pathlib,subprocess,shutil
r=pathlib.Path({q.REMOTE!r})/{q.RUN!r};cg=pathlib.Path('/sys/fs/cgroup/memory')
out=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),disk_free=shutil.disk_usage(r).free,
 memory_cap=int((cg/'memory.limit_in_bytes').read_text()),memory_used=int((cg/'memory.usage_in_bytes').read_text()),
 GPU=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True))
for name in ('STATE.json','SOURCE_COMPLETE.json','TARGET_COMPLETE.json','SOURCE_PREFLIGHT.json','TARGET_PREFLIGHT.json'):
 if (r/name).exists():out[name]=json.loads((r/name).read_text())
for prefix in ('SOURCE_PREFLIGHT','SOURCE_TRAIN','TARGET_PREFLIGHT','TARGET_RUN'):
 for suffix in ('FAILURE.json',):
  if (r/(prefix+'_'+suffix)).exists():out[prefix+'_'+suffix]=json.loads((r/(prefix+'_'+suffix)).read_text())
 if (r/(prefix+'_LAUNCH.json')).exists():
  pid=json.loads((r/(prefix+'_LAUNCH.json')).read_text())['pid'];path=pathlib.Path('/proc')/str(pid)/'cmdline';out[prefix+'_PROCESS']=path.read_bytes().decode().replace(chr(0),' ') if path.exists() else None
 if (r/(prefix+'.log')).exists():
  path=r/(prefix+'.log')
  with path.open('rb') as f:f.seek(max(0,path.stat().st_size-1600));out[prefix+'_LOG_TAIL']=f.read().decode(errors='replace')
print(json.dumps(out))
''')
    path=LOCAL/('STATUS_'+stamp()+'.json');dump(path,result)
    print(json.dumps(dict(path=path.relative_to(ROOT).as_posix(),**result),ensure_ascii=False))


def archive(stage):
    lock=verified();folder=LOCAL/(stage.lower()+'_final_'+stamp())
    receipt=remote(f'''
import hashlib,json,pathlib,tarfile,shutil,os,subprocess,datetime
r=pathlib.Path({q.REMOTE!r})/{q.RUN!r};stage={stage!r};read=lambda p:json.loads(p.read_text())
assert read(r/(stage+'_COMPLETE.json'))['registration_sha256']=={lock['registration_sha256']!r}
assert not any(r.glob('*_FAILURE.json'))
for prefix in (stage+'_PREFLIGHT',stage+('_TRAIN' if stage=='SOURCE' else '_RUN')):
 pid=read(r/(prefix+'_LAUNCH.json'))['pid'];proc=pathlib.Path('/proc')/str(pid)/'cmdline';assert not proc.exists() or not proc.read_bytes()
files={{}};checkpoints={{}}
for p in r.rglob('*'):
 if not p.is_file() or '__pycache__' in p.parts or p.name.startswith('archive_'):continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
  os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
 row=dict(sha256=h.hexdigest(),bytes=p.stat().st_size)
 (checkpoints if p.suffix=='.pt' else files)[p.relative_to(r).as_posix()]=row
assert sum(x['bytes'] for x in files.values())<512*1024**2
assert shutil.disk_usage(r).free>1536*1024**2
path=r/('archive_'+{folder.name!r}+'.tar.gz')
with tarfile.open(path,'x:gz') as tar:
 for name in files:tar.add(r/name,arcname=name,recursive=False)
print(json.dumps(dict(archive=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),files=files,remote_checkpoints_only=checkpoints,
 process_exit_verified=True,time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 GPU_after_exit=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True))))
''',timeout=600)
    folder.mkdir();dump(folder/'REMOTE_INVENTORY.json',receipt);scp(HOST+':'+receipt['archive'],folder/'run.tar.gz');assert sha(folder/'run.tar.gz')==receipt['sha256']
    run=folder/'run';run.mkdir()
    with tarfile.open(folder/'run.tar.gz','r:gz') as tar:
        for item in tar.getmembers():assert item.isfile() and not item.issym() and not item.islnk() and (run/item.name).resolve().is_relative_to(run.resolve())
        tar.extractall(run,filter='data')
    for name,item in receipt['files'].items():assert sha(run/name)==item['sha256']
    path=LOCAL/(stage+'_ARCHIVE_RECEIPT.json');dump(path,dict(path=folder.relative_to(ROOT).as_posix(),archive_sha256=receipt['sha256']))
    update(dict(status=stage.lower()+'_archived_pending_verification',**{stage.lower()+'_archive':folder.relative_to(ROOT).as_posix()}),
        'SemanticHalf_three_seed_'+stage.lower()+'_archived',path);print(json.dumps(read(path)))
def upload_source_proof():
    proof=LOCAL/'SOURCE_LOCAL_VERIFICATION.json';assert read(proof)['passed'];verified()
    remote(f"import pathlib,json;p=pathlib.Path({q.REMOTE!r})/{q.RUN!r}/'SOURCE_LOCAL_VERIFICATION.json';assert not p.exists();print(json.dumps(dict(absent=True)))")
    scp(proof,HOST+':'+q.REMOTE+'/'+q.RUN+'/SOURCE_LOCAL_VERIFICATION.json')
    result=remote(f"import pathlib,json,hashlib;p=pathlib.Path({q.REMOTE!r})/{q.RUN!r}/'SOURCE_LOCAL_VERIFICATION.json';assert hashlib.sha256(p.read_bytes()).hexdigest()=={sha(proof)!r};print(json.dumps(dict(passed=True)))")
    dump(LOCAL/'SOURCE_PROOF_UPLOAD.json',result);print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('register','upload','source-preflight','train','target-preflight','target','status','archive-source','archive-target','upload-source-proof','tests'));action=parser.parse_args().action
    try:
        if action in ('source-preflight','train','target-preflight','target'):launch({'source-preflight':'SOURCE_PREFLIGHT','train':'SOURCE_TRAIN','target-preflight':'TARGET_PREFLIGHT','target':'TARGET_RUN'}[action])
        elif action.startswith('archive-'):archive(action.split('-')[1].upper())
        elif action=='upload-source-proof':upload_source_proof()
        elif action=='tests':print(json.dumps(tests()))
        else:globals()[action]()
    except Exception:
        import traceback
        dump(LOCAL/('ACTION_FAILURE_'+action+'_'+stamp()+'.json'),dict(time=stamp(),traceback=traceback.format_exc()));raise
