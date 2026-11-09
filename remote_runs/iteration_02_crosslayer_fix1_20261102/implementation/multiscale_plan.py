"""Standard-library V18 protocol/statistics. No local model execution."""
from collections import defaultdict
from pathlib import Path
import sys

reference=Path(__file__).parent/'reference'
if not reference.is_dir():reference=Path(__file__).parent.parent/'v17_patch_forensics'
sys.path.insert(0,str(reference))
from patch_plan import read,digest,fingerprint,require,verify_split,auc,paired,ANCHOR_SHA,REFERENCE_SHA

RUN_NAME='v18_multiscale_ir_20261101'
REMOTE_ROOT='/root/autodl-tmp/drf_dual_transfer_20261026'
ARMS=('DeepOnlyAdapter','ShallowMultiScale')
SEED=2027


def configuration():
    return dict(seed=SEED,arms=list(ARMS),epochs=10,batch_size=16,workers=2,
        model='V5 IR_LayerScale with a zero-initialized residual scale adapter; g=1',
        shared_semantic_sha256=ANCHOR_SHA,reference_sha256=REFERENCE_SHA,
        taps_after_encoder_indices=[2,4,6],tap_grids=[224,112,56],fusion_grid=16,hidden=128,
        route_control=['deep','deep','deep'],route_candidate=['shallow','middle','deep'],
        adapter='deep + Conv1x1(cat(non-affine GroupNorm8(pooled taps)))',
        adapter_added_parameters=49152,adapter_initialization='all-zero; original RNG preserved',
        same_parameter_count=True,same_function_space_claimed=False,same_flops_claimed=False,
        objective='unchanged V5 two-view BCE(s+delta)+0.1 BCE(r), plus0.05 delta MSE',
        auxiliary_patch_weighting=False,gray_input=False,gate_training=False,
        primary_sampling='unchanged V5 method-balanced replacement, seed+10000*epoch',
        endpoint_rule='earliest best full source VAL AUC among epochs0..10; also fixed epoch10',
        source_val_videos=690,source_val_frames=22080,train_videos=2873,train_frames=11492,
        source_test_access=False,target_inference=False,target_training=False,
        read_prior_target_manifests_for_scoped_cache_lifecycle_only=True,
        degraded_source='fixed resize192 to224, blur0.35, JPEG85; not an external domain',
        pilot_signal='clean candidate-control>0; clean>=95%; clean-oldIR>=-0.02pp; degraded candidate-control>=0.10pp and descriptive lower bound>0',
        automatic_expansion=False,promotion=False,semantic_retraining=False,
        original_residual_training_repeated=False,memory_reserve_gib=6,disk_reserve_gib=1)


def adapter_shape(channels=128):
    require(channels>0 and channels%8==0,'GroupNorm geometry')
    return (channels,3*channels,1,1)


def route_indices(arm):
    require(arm in ARMS,'Unknown arm')
    return (2,2,2) if arm==ARMS[0] else (0,1,2)


def merge_videos(columns):
    """Align frozen/new predictions by identity; do not depend on saved file order."""
    merged={}
    for name,rows in columns.items():
        seen=set()
        for row in rows:
            key=row['video_id'];require(key not in seen,'Duplicate source video');seen.add(key)
            meta={field:row[field] for field in ('video_id','family_id','label','frames')}
            out=merged.setdefault(key,dict(**meta,scores={}))
            require(all(out[field]==value for field,value in meta.items()),'Source identity changed')
            out['scores'][name]=row['score']
    require(len(merged)==690 and all(row['frames']==32 and len(row['scores'])==len(columns) for row in merged.values()),'Incomplete source denominator')
    return [merged[key] for key in sorted(merged)]


def statistics(videos,scope):
    names=list(videos[0]['scores'])
    columns={name:[row['scores'][name] for row in videos] for name in names}
    labels=[row['label'] for row in videos]
    results={name:auc(labels,score) for name,score in columns.items()}
    comparisons={name:{base:paired(videos,columns[base],columns[name],scope=scope)
                       for base in ('Semantic','OldV2','ReusedV5IR')} for name in ARMS}
    comparisons[ARMS[1]][ARMS[0]]=paired(videos,columns[ARMS[0]],columns[ARMS[1]],scope=scope)
    fixed=paired(videos,columns[ARMS[0]+'_epoch10'],columns[ARMS[1]+'_epoch10'],scope=scope+'; fixed epoch10')
    return dict(aucs=results,paired=comparisons,fixed_epoch10_pair=fixed,videos=len(videos),
                frames=sum(row['frames'] for row in videos),real_videos=labels.count(0),fake_videos=labels.count(1))


def pilot_signal(clean,degraded):
    c,r=ARMS[1],ARMS[0]
    return bool(clean['aucs'][c]>=.95 and clean['aucs'][c]-clean['aucs']['ReusedV5IR']>=-.0002
        and clean['paired'][c][r]['net_pp']>0 and degraded['paired'][c][r]['net_pp']>=.10
        and degraded['paired'][c][r]['descriptive_95_ci_pp'][0]>0)
