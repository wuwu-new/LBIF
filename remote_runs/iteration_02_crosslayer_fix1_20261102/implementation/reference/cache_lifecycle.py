"""Standard-library provenance and scoped read-cache lifecycle for V17 fix4."""
from array import array
import json
import math
import os
from pathlib import Path
import stat
from collections import Counter

RUN_NAME='v17_patch_forensics_eval_fix4_20261101'
PARENT_NAME='v17_patch_forensics_20261101'
PREVIOUS_NAME='v17_patch_forensics_eval_fix3_20261101'
MODELS=('OldV2','ReusedV5IR','UniformPatch','SoftForensics')
COMPLETED=('cdf1','cdf2','dfd','dfdc','dfdcp')


def release_read_cache(path,allowed,adviser=None):
    """Read-only per-file kernel advice, never unlink/truncate or global reclaim."""
    name=str(path)
    if name not in allowed:raise ValueError('Path outside verified consumed/read manifest')
    if Path(name).is_symlink():raise ValueError('Symlink is not an authorized cache-advice target')
    fd=os.open(name,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):raise ValueError('Expected regular image file')
        function=adviser if adviser is not None else os.posix_fadvise
        function(fd,0,0,getattr(os,'POSIX_FADV_DONTNEED',4))
        after=os.fstat(fd)
        if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
            raise ValueError('Input file changed during read-cache advice')
        return before.st_size
    finally:os.close(fd)


def validate_frame(row,original,index,models=MODELS):
    if row['evaluation_row_index']!=index:raise ValueError('Noncontiguous prefix index')
    for field in ('image_path','video_id','family_id','label'):
        if row[field]!=original[field]:raise ValueError('Frame/manifest binding changed: '+field)
    if row['source_frame_number']!=original.get('source_frame_number'):raise ValueError('Invented source frame')
    if row['source_frame_number_known']!=('source_frame_number' in original):raise ValueError('Source-frame provenance changed')
    if set(row['scores'])!=set(models):raise ValueError('Incomplete model scores')
    for values in row['scores'].values():
        if set(values)!=set(('s','d','r')) or not all(math.isfinite(value) for value in values.values()):raise ValueError('Invalid saved logits')


def load_prefix(file,manifest,expected_count=None):
    predictions={name:{key:array('f') for key in ('s','d','r')} for name in MODELS}
    count=0
    with Path(file).open(encoding='utf-8') as handle:
        for index,line in enumerate(handle):
            if index>=len(manifest):raise ValueError('Prefix longer than manifest')
            row=json.loads(line)
            validate_frame(row,manifest[index],index)
            for name,values in row['scores'].items():
                for key,value in values.items():predictions[name][key].append(value)
            count+=1
    if expected_count is not None and count!=expected_count:raise ValueError('Prefix count changed')
    if not count or count%32 or count%16:raise ValueError('Prefix not aligned to historical video/batch boundary')
    groups=Counter(row['video_id'] for row in manifest[:count])
    if any(value!=32 for value in groups.values()) or any(row['video_id'] in groups for row in manifest[count:]):
        raise ValueError('Prefix splits a video across resumed boundary')
    if len({row['image_path'] for row in manifest[:count]})!=count:raise ValueError('Duplicate prefix image path')
    return count,predictions


def audit_previous(previous,parent,p,old):
    lock=p.read(previous/'REGISTRATION.json')
    p.require(lock['registration_sha256']==p.fingerprint({k:v for k,v in lock.items() if k!='registration_sha256'}),'Previous registration changed')
    for name,sha in lock['bundle_sha256'].items():p.require(p.digest(previous/name)==sha,'Previous bundle changed')
    p.require(p.read(previous/'RUN_FAILURE.json')['error']=="ValueError('6GiB cgroup reserve')",'Different previous failure')
    training=old.audit_training(parent,p)
    p.require(training==p.read(previous/'TRAINING_AUDIT.json'),'Training changed after previous evaluation')
    frozen=p.read(previous/'EVAL_LOCK.json')
    p.require(frozen['candidates']==training['candidates'] and frozen['terminals']==training['terminals'],'Frozen weights changed')
    p.require(not frozen['target_selection'] and frozen['constant_g']==1,'Changed target protocol')
    metrics_files={path.name[:-len('_METRICS.json')] for path in (previous/'evaluation').glob('*_METRICS.json')}
    p.require(metrics_files==set(COMPLETED),'Unexpected completed datasets; do not repeat completed work')
    receipts={}
    for name in COMPLETED:
        item=p.read(previous/'evaluation'/f'{name}_METRICS.json')
        videos=p.read(previous/'evaluation'/f'{name}_VIDEOS.json')
        reference=frozen['manifests'][name]
        p.require(len(videos)==reference['videos']==item['videos'] and item['frames']==reference['frames'],'Completed denominator changed')
        for key,value in item['aucs'].items():p.require(abs(p.auc([row['label'] for row in videos],[row['scores'][key] for row in videos])-value)<1e-12,'Completed target AUC changed')
        p.require(p.read(previous/'evaluation'/f'{name}_REFERENCE_REPRODUCTION.json')['passed'],'Completed reference failed')
        receipts[name]={path.name:p.digest(path) for path in (previous/'evaluation').glob(name+'_*') if path.is_file()}
    source={path.name:p.digest(path) for path in (previous/'evaluation').glob('clean_*') if path.is_file()}
    source.update({path.name:p.digest(path) for path in (previous/'evaluation').glob('degraded_*') if path.is_file()})
    for view in ('clean','degraded'):p.require((previous/'evaluation'/f'{view}_SUMMARY.json').is_file(),'Source not complete')
    wild=frozen['manifests']['wild']
    p.require(p.digest(wild['manifest'])==wild['manifest_sha256'],'Wild manifest changed')
    rows=p.read(wild['manifest'])
    prefix=previous/'evaluation/wild_FRAMES.jsonl'
    count,_=load_prefix(prefix,rows)
    p.require(count==20480 and len(rows)==25792,'Unexpected prefix; preserve without guessing')
    p.require(not (previous/'evaluation/uadfv_FRAMES.jsonl').exists(),'Unexpected UADFV work; do not repeat')
    return dict(previous_registration_sha256=lock['registration_sha256'],previous_eval_lock_sha256=p.digest(previous/'EVAL_LOCK.json'),
        previous_failure_sha256=p.digest(previous/'RUN_FAILURE.json'),completed_target_receipts=receipts,
        source_receipts=source,wild_prefix=dict(path=str(prefix),sha256=p.digest(prefix),bytes=prefix.stat().st_size,
        frames=count,total=len(rows),new_frames=len(rows)-count),remaining_datasets=['wild','uadfv'],
        new_inference_frames=8160,training=training,training_repeated=False)


def consumed_paths(previous,audit,p):
    reference=p.read(previous/'EVAL_LOCK.json')['manifests']
    seen=set()
    for name in (*COMPLETED,'wild'):
        item=reference[name]
        p.require(p.digest(item['manifest'])==item['manifest_sha256'],'Manifest changed before scoped cache advice')
        rows=p.read(item['manifest'])
        count=audit['wild_prefix']['frames'] if name=='wild' else len(rows)
        for row in rows[:count]:
            path=row['image_path']
            if path not in seen:
                seen.add(path)
                yield path
