"""Selected SemanticHalf, two new independent correction seeds; stdlib only."""
import hashlib
import json
import math
import struct

REMOTE='/root/autodl-tmp/drf_dual_transfer_20261026'
RUN='semantic_half_three_seed_fix1_20261104'
PARENT='iteration_02_crosslayer_fix1_20261102'
PARENT_REG='fd3b984647210e456cb0000291cd76d592a7e04e7c7c84421f91e5c862571b5f'
HISTORY='iteration_03_sequence_target_20261102'
ARMS=('CrossLayerNoImage',)
SEEDS=(2027,3407,4099)
NEW_SEEDS=(3407,4099)
DOMAINS=('cdf1','cdf2','dfd','dfdc','dfdcp','wild','uadfv')
COUNTS=((32,57),(156,326),(138,1692),(1538,1800),(107,232),(396,410),(44,45))


def fingerprint(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def sigmoid(x):return 1/(1+math.exp(-x)) if x>=0 else math.exp(x)/(1+math.exp(x))


def configuration():
    return dict(winner='SemanticHalf',seeds=list(SEEDS),new_training_seeds=list(NEW_SEEDS),reuse_seed2027=True,
        compatibility_repair='per-tensor CUDA safe_open device0 pread; failed parent288ffb no forward/update',
        architecture='exact historical CrossLayerNoImage, layers8/16/24, IR_LayerScale+ShallowMultiScale',
        inherited_frozen_anchor_seed=2027,token_input_fixed_initialization_seed=2202027,
        trainable_scope='independent token correction/readout; not fresh CLIP or historical semantic anchor training',
        epochs=10,updates_per_epoch=719,updates_per_new_seed=7190,new_retained_updates=14380,
        source_preflight_disposable_updates_per_invocation=4,source_total_disposable_updates=8,
        source_train_frames=11492,source_train_videos=2873,source_train_families=291,
        source_val_frames=22080,source_val_videos=690,source_val_families=70,
        target_frames=223136,target_videos=6973,new_target_model_frame_forwards=446272,
        source_probe_frames=64,target_probe_frames_per_domain=16,source_probe_starts=[0,5504,11040,16560],
        source_preflight_predict_forwards_per_invocation=192,target_preflight_predict_forwards_per_invocation=336,
        retained_training_frame_forwards=459680,source_VAL_frame_forwards=485760,checkpoint_replay_frame_forwards=352,
        source_fresh_CLIP_frames=23048,target_fresh_CLIP_frames=223360,
        batch=16,workers=2,lr=.0003,weight_decay=.01,auxiliary_BCE=.1,delta_consistency_MSE=.05,clip_norm=1.,
        bootstrap=1000,bootstrap_seed_start=20271104,
        sampling='WeightedRandomSampler original method weights/replacement; generator seed+10000*epoch',
        augmentation='exact fixed per-path nuisance-v1 two views; original4 TRAIN frames; unchanged across seeds',
        training_logits='full s+delta; never half-scale in training',endpoint=10,coefficient=.5,threshold=.5,
        evaluation='FP32 logit then FP32 half; mean32 stdlib double sigmoid probabilities',
        source_evaluations=list(range(11)),checkpoint_all_epochs=True,source_TEST=False,target_masks=False,
        target_labels_for_training=False,coefficient_search=False,seed_search=False,epoch_search=False,
        cache='two readonly shared RAM TRAIN views once; prior source VAL_EARLY+VAL_DISK verified reuse; target streaming fresh CLIP',
        output_allowance_bytes=2*1024**3,memory_reserve_bytes=6*1024**3,disk_reserve_bytes=1024**3,
        adaptive_selection=True,all_three_are_fresh_independent_selection=False,new_two_report_separately=True,
        automatic_extra_seeds=False,old_tasks_paused=True,complete_optimization_round=False)
