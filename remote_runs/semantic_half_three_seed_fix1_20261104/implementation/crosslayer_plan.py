"""Standard-library protocol for bounded round02; no local model imports."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if not (HERE / 'frame_plan.py').exists():
    sys.path.insert(0, str(HERE.parent / 'v22_frame_coverage_fix1'))
from frame_plan import read, digest, fingerprint, require, verify_split, auc, paired

REMOTE_ROOT = '/root/autodl-tmp/drf_dual_transfer_20261026'
RUN_NAME = 'iteration_02_crosslayer_fix1_20261102'
ARMS = ('LastLayerResidual', 'CrossLayerResidual', 'CrossLayerNoImage')
LAYERS = (8, 16, 24)
SEED = 2027
GIB = 1024**3
OLD_VAL_RUN = 'v22_frame_coverage_fix1_20261101'
CLIP_SOURCE_SHA = 'ee6c3996aa6c25c9051bba1961200e58fcdc3ffda7f83120e83e63b2194ff3c1'


def routes(arm):
    require(arm in ARMS, 'Unknown crosslayer arm')
    return (2, 2, 2) if arm == ARMS[0] else (0, 1, 2)


def cache_estimate(train_frames=11492, val_frames=22080):
    require(type(train_frames) is int and type(val_frames) is int and train_frames > 0 and val_frames > 0,
            'Invalid cache denominator')
    return dict(two_train_views_bytes=2 * train_frames * (3 * 224 * 224 + 3 * 256 * 192 * 2 + 4),
                added_val_bytes=val_frames * 2 * 256 * 192 * 2,
                old_val_reused_bytes=val_frames * (3 * 224 * 224 + 256 * 192 * 2 + 4),
                source_output_allowance_bytes=2 * GIB,
                future_remaining_same_scale_output_estimate_bytes=8 * GIB,
                disk_reserve_bytes=GIB, memory_reserve_bytes=6 * GIB)


def configuration():
    return dict(iteration=2, arms=list(ARMS), seed=SEED, epochs=10,
                implementation_repair=dict(failed_parent='iteration_02_crosslayer_20261102',
                    failed_parent_registration_sha256='94e5a55639f15c83768000b4f400ba830323b57a09901ba418a3e32f410bea91',
                    diagnosis_result_sha256='49be7575e34aead61cc38c557e4ee01caa3390b9d12473a033865f121d4e0c95',
                    change='cast full257 tokens before patch slicing; owned guard binding immediately after CUDA creation',
                    scientific_protocol_unchanged=True, failure_overwritten=False, counts_as_new_round=False),
                primary_endpoint=dict(epoch=10, g=1),
                secondary_endpoint='earliest best source VAL0..10, descriptive only, never target checkpoint',
                layers=list(LAYERS), layer_indexing='hidden_states[k] after kth transformer block; 0 is pre-encoder',
                backbone='pinned frozen CLIP ViT-L/14,24 blocks,257x1024 output at224',
                cache_projection='same frozen historical 1024to192 semantic_query for each layer; fp16 patches',
                semantic_exit='complete historical head on original last_hidden_state; fp32 logit unchanged',
                readout='non-affine LayerNorm192 per route; Linear576to192 biasFalse initialized [I,I,I]/3',
                full_model='V18 ShallowMultiScale plus same crosslayer readout feeding existing local interaction',
                last_layer_control_routes=[24, 24, 24], cross_layer_routes=[8, 16, 24],
                no_image_control='independently trained same readout/fusion/width/depth; Linear192to32 token input; same residual encoder blocks on16grid without pooling; no RGB/SRM access',
                no_image_parameter_delta_vs_full=2272,
                same_full_arm_parameter_count=True, identical_function_space_or_FLOPs_claimed=False,
                full_arm_common_initialization=True, shared_training_prefix=False,
                original_independent_first_epoch_replay_passed=False,
                original_failure_not_retested_or_overwritten=True,
                training_videos=2873, training_families=291, training_frames=11492,
                source_val_videos=690, source_val_frames=22080,
                batch_size=16, workers=2, updates_per_epoch=719, per_arm_updates=7190,
                total_new_optimizer_updates=3 * 7190,
                sampler='unchanged method-balanced replacement,2027+10000*epoch,paired slot SHA',
                augmentation='unchanged original four frames and path-fixed resize[.75,1],blur[0,.4],JPEG[65,95]',
                objective='unchanged mean two-view BCE(s+delta)+0.1BCE(aux) plus0.05 delta MSE; AdamW3e-4,wd.01,clip1',
                no_image_auxiliary='same0.1 BCE on token-derived auxiliary head; not a zeroed frozen Full ablation',
                cache='two RAM TRAIN views; new VAL8/16 projections only on disk; old VAL RGB/24/logit verified reused',
                resource_estimate=cache_estimate(), changed_semantic_or_pixels_require_fresh_encoding=True,
                source_gain_execution_gate=False,
                targets='fixed all three epoch10 weights,complete7domains6973videos223136frames,32mean probabilities',
                layer_or_target_g_or_checkpoint_scan=False, adaptive_exploration=True,
                seed_training_sd='NA', source_test_access=False, target_labels_for_training=False,
                promotion=False, formal_default='OldV2', research_reference='HistoricalV18Multi',
                decision='retain crosslayer route only if macro exceeds matched LastLayerResidual; residual credit only vs CrossLayerNoImage, otherwise semantic-readout benefit only',
                invalid_or_hard_sample_filtering=False)


def source_summary(columns):
    require(set(columns) == set(ARMS), 'Missing source arm')
    base = columns[ARMS[0]]
    keys = [row['video_id'] for row in base]
    require(len(keys) == len(set(keys)) == 690, 'Wrong VAL video denominator')
    require(all(row['frames'] == 32 for row in base), 'Wrong VAL frame denominator')
    labels = [row['label'] for row in base]
    require(labels.count(0) == 138 and labels.count(1) == 552 and len({r['family_id'] for r in base}) == 70,
            'Wrong VAL class/family coverage')
    scores = {}
    for arm, rows in columns.items():
        mapping = {r['video_id']: r for r in rows}
        require(len(rows) == len(mapping) == 690 and set(mapping) == set(keys), 'Unaligned source arm')
        for row in base:
            other = mapping[row['video_id']]
            require(all(other[k] == row[k] for k in ('label', 'frames', 'family_id')), 'Changed source metadata')
            require(abs(other['semantic'] - row['semantic']) < 2e-7, 'Shared semantic changed')
        scores[arm] = [mapping[key]['score'] for key in keys]
    scope = 'adaptive source VAL; single residual seed; independently trained no-image control; SD=NA'
    return dict(videos=690, frames=22080, real_videos=138, fake_videos=552, families=70,
                aucs={arm: auc(labels, values) for arm, values in scores.items()},
                paired={base_arm: paired(base, scores[base_arm], scores[ARMS[1]], scope=scope)
                        for base_arm in (ARMS[0], ARMS[2])})
