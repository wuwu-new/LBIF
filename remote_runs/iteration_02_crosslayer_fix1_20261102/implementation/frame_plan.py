"""Standard-library protocol for the matched V18 frame-coverage pilot."""
from collections import defaultdict
from pathlib import Path
import sys

reference=Path(__file__).parent/'reference'
if not reference.is_dir():reference=Path(__file__).parent.parent/'v17_patch_forensics'
sys.path.insert(0,str(reference))
from patch_plan import read,digest,fingerprint,require,verify_split,auc,paired,ANCHOR_SHA,REFERENCE_SHA

RUN_NAME='v22_frame_coverage_fix1_20261101'
ORIGINAL_V22='v22_frame_coverage_20261101'
REUSED_POOL_SHA='f419d6c46d9ad8f758d96790afd3fdbed3270d9c6b5a497cd940beef6df60535'
PARENT_NAME='v18_multiscale_ir_20261101'
REMOTE_ROOT='/root/autodl-tmp/drf_dual_transfer_20261026'
INDEX='/root/autodl-tmp/results/protocol1/unified_indexes_ffpp_c23/ffpp_c23_train.json'
INDEX_SHA='f57f61505262950a623f22f144a9f3c80d04f368c613b66cf7e187f1f5d558c1'
ARMS=('FixedFour','RotatingFour')
SEED=2027


def configuration():
    return dict(seed=SEED,arms=list(ARMS),epochs=10,batch_size=16,workers=2,
        model='unchanged V18 ShallowMultiScale IR_LayerScale; shared historical semantic; g=1',
        shared_semantic_sha256=ANCHOR_SHA,reference_sha256=REFERENCE_SHA,
        training_videos=2873,training_families=291,slots_per_epoch=11492,updates_per_epoch=719,
        pool_frames=91936,source_val_videos=690,source_val_frames=22080,
        schedule='32 ordered frames in four quartiles of eight; historical frame first, other seven ordered by SHA256(seed,video,slot,path); epochs9/10 repeat1/2',
        same_initialization=True,same_slot_sampling=True,first_epoch_identical=True,
        augmentation='unchanged path-fixed V18 clean/resize-blur-JPEG; not refreshed by epoch',
        cache='disk-streamed compact VAL + current epoch two TRAIN RAM views only; cache building restores all RNG states',
        objective='unchanged V5 mean two-view BCE(s+delta)+0.1 BCE(r), plus0.05 delta MSE',
        endpoint_rule='primary fixed epoch10; secondary earliest best full source VAL among epochs0..10',
        diagnostic_signal='fixed epoch10 rotating-minus-fixed>0 and rotating>=95% source VAL; exploratory only',
        equal_optimizer_updates=True,equal_total_compute_claimed=False,
        historical_training_repeated=False,new_path_matching_control=True,
        semantic_retraining=False,gate_training=False,source_test_access=False,target_inference=False,
        automatic_expansion=False,promotion=False,memory_reserve_gib=6,disk_reserve_gib=1)


def build_pool(train,index):
    selected={row['video_id'] for row in train};groups=defaultdict(list)
    for row in index:
        if row['video_id'] in selected:groups[row['video_id']].append(row)
    require(set(groups)==selected,'Incomplete TRAIN video pool')
    pools={};slots={}
    for i,row in enumerate(train):
        key=(row['video_id'],row['frame_index'])
        require(key not in slots and row['frame_index'] in range(4),'Duplicate/invalid historical slot')
        slots[key]=i
    require(len(slots)==4*len(selected),'Unequal historical slots')
    for vid,rows in sorted(groups.items()):
        ordered=sorted(rows,key=lambda row:row['source_frame_number'])
        require(len(ordered)==32 and len({r['image_path'] for r in ordered})==32,'TRAIN pool must have32 unique frames')
        require(len({r['source_frame_number'] for r in ordered})==32,'Repeated source frame number')
        pools[vid]=[]
        for slot in range(4):
            old=train[slots[vid,slot]];quarter=ordered[8*slot:8*slot+8]
            require(all(r['label']==old['label'] for r in quarter),'Pool label changed')
            historical=[r for r in quarter if r['image_path']==old['image_path']]
            require(len(historical)==1 and historical[0]['source_frame_number']==old['source_frame_number'],'Historical frame outside its quartile')
            remaining=sorted((r for r in quarter if r['image_path']!=old['image_path']),
                key=lambda r:fingerprint([SEED,vid,slot,r['image_path']]))
            pools[vid].append([dict(old,image_path=r['image_path'],source_frame_number=r['source_frame_number'],
                sha256=old['sha256'] if r['image_path']==old['image_path'] else None) for r in historical+remaining])
    return pools


def epoch_rows(train,pools,arm,epoch):
    require(arm in ARMS and epoch in range(1,11),'Invalid arm/epoch')
    if arm=='FixedFour':return train
    return [pools[row['video_id']][row['frame_index']][(epoch-1)%8] for row in train]


def summary(columns):
    rows=columns['FixedFour_epoch10'];keys=[row['video_id'] for row in rows]
    require(len(keys)==len(set(keys))==690,'Wrong VAL denominator')
    scores={}
    for name,values in columns.items():
        mapping={r['video_id']:r for r in values}
        require(len(values)==len(mapping)==690 and set(mapping)==set(keys),'Unaligned VAL scores')
        for row in rows:
            other=mapping[row['video_id']]
            require(all(other[k]==row[k] for k in ('label','family_id','frames')) and other['frames']==32,'VAL identity changed')
        scores[name]=[mapping[key]['score'] for key in keys]
    labels=[row['label'] for row in rows]
    comparisons={}
    scope='reused source VAL; one shared semantic and paired residual seed; exploratory, no multiplicity correction'
    for ending in ('epoch10','best'):
        base='FixedFour_'+ending;candidate='RotatingFour_'+ending
        comparisons[ending]=paired(rows,scores[base],scores[candidate],scope=scope)
    return dict(aucs={name:auc(labels,values) for name,values in scores.items()},paired=comparisons,
        videos=690,frames=22080,families=len({r['family_id'] for r in rows}),
        real_videos=labels.count(0),fake_videos=labels.count(1))
