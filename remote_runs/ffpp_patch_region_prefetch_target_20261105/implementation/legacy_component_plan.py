"""Finite whole-method/component experiment. Standard library only."""
import collections
from fractions import Fraction
import hashlib
import json
import math
import random
import struct

REMOTE = '/root/autodl-tmp/drf_dual_transfer_20261026'
RUN = 'ffpp_paper_components_20261104'
SOURCE = 'ffpp_legacy_delta_full_fix2_20261104'
SOURCE_REG = '4e7ce9ae6ac3e109e08ae67a63f0a98a20bb1546618ae9406030b6cea19d1d85'
TARGET = 'ffpp_legacy_delta_sourcebest_target_20261104'
TARGET_REG = '5878f9871cf5c27d993878f54afaf48fc4b9bd3fa1dcacf3fdb2733cf80a8c0e'
SELECTION = 'ffpp_legacy_delta_sourcebest_20261104'
NEW_ARMS = ('UniformNoAux', 'UniformBoundary', 'RGBBoundary')
SEEDS = (2027, 3407, 4099)
METHODS = ('real', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures')
DOMAINS = ('cdf1', 'cdf2', 'dfd', 'dfdc', 'dfdcp', 'wild', 'uadfv')
COUNTS = ((32,57), (156,326), (138,1692), (1538,1800), (107,232), (396,410), (44,45))
# Preserve the imported, archived model ABI, not extra trainable arms.
ARMS = ('LegacyPoolDelta', 'NormalizedPoolDelta')
BOUND, PENALTY, EPS = .5, .01, 1e-6
PROBES = tuple(i for start in (0,5504,11040,16560) for i in range(start,start+16))
GIB = 1024**3

def fingerprint(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def f32(x): return struct.unpack('<f',struct.pack('<f',x))[0]
def identity(r): return r['video_id'],r['source_frame_number']
def sigmoid(x): return 1/(1+math.exp(-x)) if x>=0 else math.exp(x)/(1+math.exp(x))

def variant(arm):
    assert arm in NEW_ARMS
    return dict(input='RGB' if arm=='RGBBoundary' else 'signed_RGB_minus_Gaussian5_sigma1',
                pooling='predicted_boundary_cells_quarter_floor' if arm=='RGBBoundary' else 'uniform',
                boundary_coefficient=0. if arm=='UniformNoAux' else .2)

def configuration():
    return dict(primary_question='whole dual-stream vs same-protocol strong semantic branch; paired internal component ablations',
        new_arms={a:variant(a) for a in NEW_ARMS},seeds=list(SEEDS),epochs=10,updates_per_epoch=719,batch=16,
        retained_updates=64710,retained_training_frame_forwards=1035360,
        preflight_disposable_updates=18,runtime_disposable_updates=0,automatic_retries=0,
        source_VAL_frame_forwards=1987200,selected_localization_forwards=132480,selected_reload_forwards=576,
        source_preflight_anchor_forwards=192,source_preflight_synthetic_forwards=72,
        source_preflight_total_forwards=264,target_preflight_forwards_per_invocation=2592,
        full_target_residual_forwards=2008224,target_preflight_total_forwards=5184,
        source_train_frames=11492,source_train_videos=2873,source_train_families=291,
        source_val_frames=22080,source_val_videos=690,source_val_families=70,
        target_frames=223136,target_videos=6973,
        sampling='Random(seed+100003*epoch);719 replacement batches;8real+2eachfake',
        optimizer=dict(lr=.0003,weight_decay=.01,betas=[.9,.999],eps=1e-8,clip_norm=1.),
        augmentation='none; exact existing aligned residual pixels; semantic scalar reuse only',
        source_input_integrity='every actual image/mask read SHA checked before identical legacy decoding',
        architecture='same inherited16/32/64 CNN/full256 decoder, constructor RNG, biasfree zero64to1 head',
        delta='.5*tanh(head(pooled)); .01 mean(delta**2)',
        semantic='same-seed verified frozen CrossLayerNoImage epoch10, no new semantic encoding/training',
        fusion='f32(f32(s_seed/2).detach()+delta)',
        inactive_decoder='UniformNoAux computes decoder; zero-connected auxiliary; no supervised decoder gradients; allocated not effective capacity matched',
        selection='per-arm per-seed complete source VAL video AUC epoch1..10; exact ranks; ties earliest',
        all_epochs_saved=True,epoch10_source_reported=True,target_selection=False,
        localization='selected UniformBoundary/RGBBoundary only;eligible VAL masks;truth>0.1,pred>=0.5,2pixel tolerance;not selection',
        aggregation='original-order sequential FP64 sum32 sigmoid probabilities divided32',
        reused_arms=['SemanticRaw','SemanticHalf','BoundaryFull'],new_CLIP_training=False,
        literature_CLIP_in_experiment=False,external_baseline_claim=False,primary_metric='video_AUC',
        supplemental_metric='frame_AUC',
        internal_metric='all seven domains equally weighted video_AUC; same-seed paired differences',
        same_seed_new_arms_initialization_exact=True,seed_search=False,coefficient_search=False,
        source_TEST=False,target_masks=False,target_labels_for_training=False,
        memory_reserve_gib=6,disk_reserve_gib=1,output_allowance_bytes=2*GIB,
        train_uint8_bytes=11492*4*256*256,bootstrap=1000,bootstrap_seed=20261104,
        whole_method_gain_not_component_attribution=True,semantic_readout_recipe_not_one_factor=True,
        full_model_posthoc_history=True,adaptive_exploration=True,independent_blind_confirmation=False,
        old_tasks_paused=True,automation_paused=True,no_new_monitor=True,counts_as_complete_round=False,deployment=False)

def sampling(rows,epoch,seed):
    assert seed in SEEDS and 1<=epoch<=10
    buckets={m:[i for i,r in enumerate(rows) if r['method']==m] for m in METHODS}
    assert all(buckets.values())
    rng=random.Random(seed+100003*epoch);result=[]
    for _ in range(719):
        batch=[]
        for m,n in zip(METHODS,(8,2,2,2,2)):batch.extend(rng.choice(buckets[m]) for _ in range(n))
        rng.shuffle(batch);result.append(batch)
    return result

def auc(y,z):
    assert len(y)==len(z)>0 and all(math.isfinite(float(x)) for x in z)
    pos=sum(y);neg=len(y)-pos
    if not pos or not neg:return None
    order=sorted(range(len(y)),key=lambda i:z[i]);rank=0;i=0
    while i<len(order):
        j=i+1
        while j<len(order) and z[order[j]]==z[order[i]]:j+=1
        rank+=(i+1+j)*sum(y[order[k]] for k in range(i,j));i=j
    return Fraction(rank-pos*(pos+1),2*pos*neg)

def videos(rows,z):
    assert len(rows)==len(z)
    groups=collections.OrderedDict()
    for r,x in zip(rows,z):
        v=groups.setdefault(r['video_id'],dict(video_id=r['video_id'],family_id=r['family_id'],label=r['label'],method=r.get('method'),frames=0,score=0.))
        assert (v['family_id'],v['label'])==(r['family_id'],r['label'])
        v['frames']+=1;v['score']+=sigmoid(float(x))
    for v in groups.values():assert v['frames']==32;v['score']/=32
    return list(groups.values())

def metrics(rows,z):
    v=videos(rows,z);y=[r['label'] for r in rows];vy=[r['label'] for r in v];a=auc(vy,[r['score'] for r in v])
    result=dict(frames=len(rows),videos=len(v),families=len({r['family_id'] for r in v}),
        real_frames=y.count(0),fake_frames=y.count(1),real_videos=vy.count(0),fake_videos=vy.count(1),
        frame_AUC=float(auc(y,z)),video_AUC=float(a),auc_fraction=[a.numerator,a.denominator],
        BCE=sum(max(t,0)-t*l+math.log1p(math.exp(-abs(t))) for l,t in zip(y,z))/len(y),
        FP=sum(r['label']==0 and r['score']>=.5 for r in v),FN=sum(r['label']==1 and r['score']<.5 for r in v))
    return result,v

def select(records):
    assert [r['epoch'] for r in records]==list(range(1,11))
    return max(records,key=lambda r:(Fraction(*r['metrics']['auc_fraction']),-r['epoch']))
