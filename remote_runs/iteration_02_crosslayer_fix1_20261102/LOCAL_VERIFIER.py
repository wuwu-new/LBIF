"""Independent saved-video-score source replay; standard library only, no checkpoint loading."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code/dual_stream_20261026/iteration02_crosslayer_fix1'))
import crosslayer_plan as p
from verify_v17_eval_fix1_archive import bundle, same


def verify(run):
    run = Path(run)
    lock = bundle(run)
    same(lock['configuration'], p.configuration())
    repair = p.read(run / 'REPAIR_EVIDENCE.json')
    scientific = p.configuration().copy()
    scientific.pop('implementation_repair')
    same(repair['parent_configuration'], scientific)
    same(repair['failed_parent_registration'], p.configuration()['implementation_repair']['failed_parent_registration_sha256'])
    same(repair['diagnostic_local_sha256'], p.configuration()['implementation_repair']['diagnosis_result_sha256'])
    p.require(repair['scientific_protocol_unchanged'] and repair['old_failure_preserved'] and repair['no_old_training_updates'] and
              repair['diagnostic']['corrected_all_probes_exact'] and repair['diagnostic']['optimizer_updates'] == 0, 'Invalid implementation repair provenance')
    same(p.read(run / 'COMPLETE.json')['final_report_sha256'], p.digest(run / 'FINAL_REPORT.json'))
    train, val = [p.read(run / 'inputs' / (name + '.json')) for name in ('fit', 'source_val')]
    same(p.verify_split(train, val), p.read(run / 'SPLIT_PROOF.json'))
    for name in ('PREFLIGHT.json', 'RUNTIME_PREFLIGHT.json'):
        probe = p.read(run / name)
        p.require(probe['passed'] and probe['models_cuda_only'] and probe['final_layer_and_complete_semantic_exact'] and
                  probe['optimizer_updates'] == 0 and probe['full_arm_initialization_equal'] and
                  not probe['shared_training_prefix'] and probe['original_independent_replay_passed'] is False, 'Invalid CUDA preflight')
        same(probe['registration_sha256'], lock['registration_sha256'])
        p.require(probe['owned_guard_bound_and_checked'] and probe['owned_nvml_pid'] > 0 and
                  probe['original_failed_preflight_preserved'], 'Missing actual CUDA ownership binding')
        same(probe['parameters'][p.ARMS[0]], probe['parameters'][p.ARMS[1]])
        same(probe['parameters'][p.ARMS[2]] - probe['parameters'][p.ARMS[1]], 2272)
        same(probe['model_initialization_sha256'][p.ARMS[0]], probe['model_initialization_sha256'][p.ARMS[1]])
        p.require(len(set(probe['common_tensor_sha256'].values())) == 1, 'Common initialization differs')
        for arm in p.ARMS:
            audit = probe['route_and_image_audits'][arm]
            p.require(audit['optimizer_updates'] == 0, 'Preflight updated weights')
            p.require((audit['intermediate_readout_sensitivity'] == 0) == (arm == p.ARMS[0]), 'Layer route mismatch')
            if arm == p.ARMS[2]:
                p.require(audit['image_independent'] and audit['image_auxiliary_gradient_l1'] == 0, 'No-image control leaked RGB')
            else:
                p.require(not audit['image_independent'] and audit['image_auxiliary_gradient_l1'] > 0, 'Full ignores image')
    for stage in ('PREFLIGHT', 'TRAIN'):
        p.require(not (run / (stage + '_FAILURE.json')).exists(), 'Failed stage cannot count as completed')
    cache = p.read(run / 'VAL_EARLY/LAYOUT.json')
    same(cache['rows'], 22080)
    same(cache['bytes'], p.cache_estimate()['added_val_bytes'])
    same(cache['layers'], [8, 16])
    same(cache['manifest_sha256'], p.fingerprint(val))
    p.require(cache['old_VAL_RGB_final_logit_exact_all_frames'] and not cache['original_cache_duplicated'] and cache['max_final_feature_error'] == 0, 'Old cache not exactly reproduced')
    report = p.read(run / 'FINAL_REPORT.json')
    same(report['registration_sha256'], lock['registration_sha256'])
    same([report['iteration'], report['per_arm_updates'], report['physical_new_updates']], [2, 7190, 21570])
    p.require(report['target_evaluation_required_even_if_source_negative'] and report['models_cuda_only'] and
              report['shared_semantic'] and report['original_independent_replay_passed'] is False and
              report['seed'] == 2027 and report['seed_standard_deviation'] is None, 'Scope/provenance changed')
    for key in ('shared_training_prefix', 'target_inference', 'source_test_access', 'target_training', 'promotion'):
        p.require(not report[key], 'Scope expanded')
    reference = {row['video_id']: row for row in p.read(run / 'BASELINE_SOURCE_VIDEOS.json')}
    meta = {}
    for row in val:
        key = row['video_id']
        out = meta.setdefault(key, dict(video_id=key, family_id=row['family_id'], label=row['label'], frames=0))
        p.require(out['family_id'] == row['family_id'] and out['label'] == row['label'], 'Inconsistent VAL manifest')
        out['frames'] += 1
    audit = p.read(run / 'CHECKPOINT_CUDA_AUDIT.json')
    p.require(audit['passed'] and audit['models_cuda_only'] and len(audit['records']) == 33, 'Missing33 CUDA checkpoint identity audits')
    lookup = {(row['identity'][0], row['identity'][1]): row for row in audit['records']}
    p.require(len(lookup) == 33, 'Duplicate checkpoint audit')
    schedules, columns, replayed = [], {}, 0
    for arm in p.ARMS:
        folder = run / 'arms' / arm
        records, schedule = [], []
        for epoch in range(11):
            row = p.read(folder / f'epoch_{epoch:02}.json')
            same([row['arm'], row['epoch'], row['seed'], row['updates'], row['optimizer_steps']], [arm, epoch, 2027, 719 if epoch else 0, epoch * 719])
            same(row['mapping_sha256'], p.fingerprint(train))
            same(row['checkpoint'], f'epoch_{epoch:02}.pt')
            same(row['video_scores'], f'epoch_{epoch:02}_VIDEOS.json')
            same(row['checkpoint_sha256'], p.digest(folder / row['checkpoint']))
            same(row['video_scores_sha256'], p.digest(folder / row['video_scores']))
            values = p.read(folder / row['video_scores'])
            p.require(len(values) == 690 and len({v['video_id'] for v in values}) == 690 and {v['video_id'] for v in values} == set(meta), 'Source video denominator changed')
            for value in values:
                same({k: value[k] for k in meta[value['video_id']]}, meta[value['video_id']])
                p.require(abs(value['semantic'] - reference[value['video_id']]['scores']['SemanticAnchor']) < 2e-7, 'Historical semantic changed')
                p.require(0 <= value['score'] <= 1, 'Invalid probability')
            same(p.auc([v['label'] for v in values], [v['score'] for v in values]), row['video_auc'])
            internal = lookup[arm, epoch]
            same(internal['identity'], [arm, epoch, 2027, lock['registration_sha256']])
            same(internal['optimizer_steps'], epoch * 719)
            same(internal['model_tensor_sha256'], row['model_tensor_sha256'])
            p.require(internal['passed'], 'CUDA checkpoint audit failed')
            if epoch:
                schedule.append(row['slot_sampling_sha256'])
                same([r['step'] for r in row['gradient_audits']], [0, 350, 718])
                p.require(all(r['semantic_frozen'] and r['gradients']['adapter'] > 0 and r['gradients']['correction'] > 0 for r in row['gradient_audits']), 'Invalid actual gradients')
            else:
                same(row['video_auc'], .9741782188615837)
                p.require(row['slot_sampling_sha256'] is None and not row['gradient_audits'], 'Epoch0 has training')
            records.append(row)
            replayed += 1
        schedules.append(schedule)
        columns[arm] = values
        best = max(records, key=lambda r: (r['video_auc'], -r['epoch']))
        same(p.read(folder / 'SELECTION.json'), best)
        same(p.read(run / 'SELECTION_LOCK.json')['secondary_selections'][arm], best)
    p.require(all(s == schedules[0] for s in schedules), 'Slot sampling differs across arms')
    same(p.read(run / 'SELECTION_LOCK.json')['primary_endpoint'], dict(epoch=10, g=1))
    p.require(not p.read(run / 'SELECTION_LOCK.json')['target_selection'], 'Target selection occurred')
    same(p.read(run / 'evaluation/ENDPOINT_COLUMNS.json'), columns)
    same(p.source_summary(columns), report['source'])
    same(report['source_goal95_met'], {a: report['source']['aucs'][a] >= .95 for a in p.ARMS})
    p.require(not any(n in sys.modules for n in ('torch', 'numpy', 'PIL')), 'Local model execution forbidden')
    return dict(passed=True, epochs_replayed=replayed, saved_video_AUC_and_paired_statistics_reproduced=True,
                source_frame_prediction_replay=False, all_checkpoint_files_sha_verified=True,
                source_videos_per_epoch=690, source_frames_per_epoch=22080,
                original_independent_replay_passed=False, shared_training_prefix=False, local_model_execution=False,
                complete_optimization_round=False, full_seven_domains_still_required=True)
