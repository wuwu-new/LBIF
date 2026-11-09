"""CUDA-only bridge study: immutable strong semantic anchor, then residual, then gate.

This is conditional transfer with three correction seeds, not three independently
trained semantic experts. Historical TRAIN families were seen by the anchor.
"""
import argparse
import copy
import gc
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import gpu_pipeline as p
import full_source_v2 as compact
import gate_study_v3 as routing
import joint_search_v4b as joint


SEEDS = [2027, 3407, 4099]
ARCHITECTURES = {k: joint.ARCHITECTURES[k] for k in
                 ['DW', 'DW_LayerScale', 'IR_LayerScale', 'ResidualRequired_LayerScale']}
SCALES = {'Fixed025': .25, 'Fixed050': .5, 'Fixed075': .75, 'NoGate': 1.}
REFERENCE = p.ROOT / 'full_source_v2/Full'
REFERENCE_SHA = 'b5c3ab1fb495f9362e4f0614826f2289f3bf7f6840351ccbeed2102ef5bc4282'
ANCHOR_SHA = 'aa7dc8cb2dc13edbc85abed47e815687ce14e9476286d02a21f5cc1b2d175865'
MIN_GAIN = .001  # 0.10 AUC percentage points, registered before new training.


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def release():
    gc.collect()
    torch.cuda.empty_cache()


def configuration():
    config = compact.configuration()
    config.update(name='strong_anchor_residual_gate_v5_20261027', seeds=SEEDS,
                  architectures=ARCHITECTURES, epochs=10,
                  architecture_selection='seed2027 nominates by source VAL g=1; all three correction seeds confirm',
                  gate_selection='after architecture lock; fixed scales or bounded soft gate; source only',
                  minimum_mean_gain_auc=MIN_GAIN, maximum_seed_regression_auc=.0005,
                  paired_bootstrap_replicates=1000,
                  training_scope='one frozen historical semantic anchor; three independent correction seeds',
                  family_scope='fit/gate families isolated for corrections; historical anchor saw original TRAIN',
                  source_val_scope='reused for epoch/model selection; descriptive, not independent confirmation',
                  targets_scope='historical target results already observed; exploratory; no target model selection',
                  old_reference_checkpoint_sha256=REFERENCE_SHA,
                  semantic_checkpoint_sha256=ANCHOR_SHA,
                  learned_gate_range=[.75, 1.], gate_steps=600,
                  resume_scope='completed stages only; partial trained folders require a separately named repair version')
    return config


def identity(config):
    files = {**p.identities(config), str(REFERENCE/'best.pt'): p.digest(REFERENCE/'best.pt')}
    for module in [sys.modules[__name__], compact, routing, joint]:
        path = Path(module.__file__).resolve()
        files[str(path)] = p.digest(path)
    if p.digest(config['semantic_checkpoint']) != ANCHOR_SHA or p.digest(REFERENCE/'best.pt') != REFERENCE_SHA:
        raise ValueError('The registered strong anchor/reference checkpoint changed')
    return files


def reference_model(config):
    state = torch.load(REFERENCE/'best.pt', map_location='cpu', weights_only=True)
    anchor = compact.CompactSemantic(config['semantic_checkpoint']).cuda()
    model = compact.CompactDual(anchor, state['spec']).cuda()
    model.load_state_dict(state['model'], strict=True)
    return model.eval().requires_grad_(False)


@torch.no_grad()
def gpu_preflight(config, root):
    rows = read(p.ROOT/'guarded_dual_20261025/run_v1/manifests/source_val.json')[:2]
    rgb = torch.stack([p.Pixels(rows)[i] for i in range(2)])
    encoder = compact.CompactEncoder(config)
    raw = compact.original_encode(encoder.clip, rgb)
    tokens = compact.compact_encode(encoder, rgb)
    expected = encoder.anchor(raw)
    torch.testing.assert_close(tokens[:, 0, 0], expected, rtol=0, atol=0)
    cache = compact.CompactFeatures(2)
    cache.patches[:] = tokens[:, 1:].cpu().half()
    cache.logits[:] = tokens[:, 0, 0].cpu()
    restored = cache[:].cuda()
    torch.testing.assert_close(tokens, restored, rtol=0, atol=0)
    anchor = compact.CompactSemantic(config['semantic_checkpoint']).cuda()
    torch.testing.assert_close(anchor(tokens), expected, rtol=0, atol=0)
    reference = reference_model(config)
    for key, value in anchor.state_dict().items():
        torch.testing.assert_close(reference.semantic.state_dict()[key], value, rtol=0, atol=0)
    checks = {}
    for name, extra in ARCHITECTURES.items():
        p.seed_all(SEEDS[0])
        model = joint.JointDual(joint.SemanticExit(anchor), {**config['model'], **extra}).cuda().eval()
        torch.testing.assert_close(model(rgb.cuda(), restored)['logit'], expected, rtol=0, atol=0)
        checks[name] = {'initial_exit_exact': True,
                       'trainable_parameters': sum(v.numel() for v in model.parameters() if v.requires_grad)}
        del model
    p.dump(root/'GPU_PREFLIGHT.json', {
        'time': p.stamp(), 'device': torch.cuda.get_device_name(),
        'raw_and_cached_semantic_logits_exact': True, 'compact_cache_roundtrip_exact': True,
        'old_reference_semantic_weights_exact': True, 'architectures': checks,
        'scope': 'contracts only; the full source reference AUC is checked before training'})
    del encoder, raw, tokens, restored, anchor, reference
    release()


def video_data(rows, pred):
    return joint.data_from_predictions(rows, pred)


def auc(data, scores):
    return routing.auc(data['y'], scores)


def probabilities(data, scale=1.):
    gate = torch.full_like(data['y'], float(scale))
    return routing.video_probabilities(data, gate)


def paired(data, candidates, reference, replicates=1000):
    """Cluster uncertainty over source families, averaged across fixed correction seeds."""
    y = data['y'].cpu().numpy()
    families = [f or v for f, v in zip(data['family_id'], data['video_id'])]
    by_family = {}
    for i, family in enumerate(families):
        by_family.setdefault(family, []).append(i)
    blocks = [np.asarray(v) for v in by_family.values()]
    scores = [s.detach().cpu().numpy() for s in candidates]
    refs = [r.detach().cpu().numpy() for r in reference]
    differences = [float(roc_auc_score(y, a)-roc_auc_score(y, b)) for a, b in zip(scores, refs)]
    if all(np.array_equal(a, b) for a, b in zip(scores, refs)):
        return {'mean_difference': 0., 'by_correction_seed': differences,
                'family_cluster_95_ci': [0., 0.], 'replicates': 0,
                'passes_registered_rule': False, 'scope': 'identical saved scores; exact zero contrast'}
    generator = np.random.default_rng(20271027)
    draws = []
    for _ in range(replicates):
        ids = np.concatenate([blocks[i] for i in generator.integers(len(blocks), size=len(blocks))])
        if len(np.unique(y[ids])) < 2:
            continue
        draws.append(float(np.mean([roc_auc_score(y[ids], a[ids])-roc_auc_score(y[ids], b[ids])
                                    for a, b in zip(scores, refs)])))
    interval = np.quantile(draws, [.025, .975]).tolist()
    return {'mean_difference': float(np.mean(differences)), 'by_correction_seed': differences,
            'family_cluster_95_ci': interval, 'replicates': len(draws),
            'passes_registered_rule': bool(np.mean(differences) >= MIN_GAIN and
                                           min(differences) >= -.0005 and interval[0] > 0),
            'scope': 'exploratory source selection; fixed anchor/seeds; no multiplicity correction'}


class BoundedGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, -4.)

    def forward(self, x):
        return 1. - .25 * self.linear(x).squeeze(-1).sigmoid()


def gate_values(data, fitted):
    x = routing.features(data, 'invariant')
    return fitted['model']((x-fitted['center'])/fitted['scale'])


def load_bounded(path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    model = BoundedGate().cuda()
    model.load_state_dict(state['state'], strict=True)
    return {'model': model.eval(), 'center': state['center'].cuda(), 'scale': state['scale'].cuda()}


def fit_bounded(fit, holdout, folder):
    folder.mkdir(exist_ok=True)
    if (folder/'Bounded.pt').exists():
        return load_bounded(folder/'Bounded.pt')
    x = routing.features(fit, 'invariant')
    center, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(.05)
    x = (x-center)/scale
    hx = (routing.features(holdout, 'invariant')-center)/scale
    p.seed_all(2027)
    model = BoundedGate().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.01)
    best, selected, history = None, None, []
    for step in range(601):
        if step:
            optimizer.zero_grad(set_to_none=True)
            loss = routing.loss_value(fit, model(x), 'rank')
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite bounded gate objective')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        if step % 10 == 0:
            with torch.no_grad():
                metrics, _ = routing.measure(holdout, model(hx))
            key = (metrics['video_auc'], -metrics['balanced_video_bce'])
            history.append({'step': step, **metrics})
            if best is None or key > best:
                best, selected = key, {'state': copy.deepcopy(model.state_dict()),
                                      'center': center.cpu(), 'scale': scale.cpu(), 'selected_step': step,
                                      'gate_range': [.75, 1.], 'fit_partition': 'gate_fit',
                                      'checkpoint_selection_partition': 'gate_holdout'}
    torch.save(selected, folder/'Bounded.pt')
    p.dump(folder/'HISTORY.json', history)
    return load_bounded(folder/'Bounded.pt')


def save_data(path, data):
    value = {k: data[k].detach().cpu().tolist() for k in ['s', 'd', 'r', 'y']}
    value.update({k: data[k] for k in ['video_id', 'family_id']})
    p.dump(path, value)


def source_data(model, folder, rows, caches, config, part='source_val'):
    path = folder/(part+'_VIDEO_DATA.json')
    if path.exists():
        value = read(path)
        return {**value, **{k: torch.tensor(value[k], device='cuda', dtype=torch.float32)
                           for k in ['s', 'd', 'r', 'y']}}
    data = video_data(rows[part], p.predict(model, caches[part], config['batch_size']))
    save_data(path, data)
    return data


def trained_model(seed, config, spec, anchor, rows, caches, folder, ident, root):
    if (folder/'SELECTION.json').exists():
        return joint.load_model(folder, anchor)
    if folder.exists():
        raise RuntimeError('Partial training folder preserved; repair in a new version: ' + str(folder))
    folder.parent.mkdir(parents=True, exist_ok=True)
    joint.status(root, 'correction_training', status='running', correction_seed=seed, group=str(folder))
    return joint.fit_residual(seed, config, spec, anchor, rows, caches, folder, ident)


def verify_saved_auc(output, names):
    result = {}
    for name in names:
        report = read(output/(name+'_METRICS.json'))
        videos = [json.loads(line) for line in (output/(name+'_VIDEOS.jsonl')).read_text().splitlines()]
        if len(videos) != report['videos'] or len({v['video_id'] for v in videos}) != len(videos):
            raise ValueError('Duplicate/missing saved evaluation videos: ' + name)
        labels = [r['label'] for r in videos]
        for group, metrics in report['models'].items():
            scores = np.asarray([v['scores'][group] for v in videos])
            if not np.isfinite(scores).all() or abs(roc_auc_score(labels, scores)-metrics['video_auc']) > 1e-10:
                raise ValueError('Saved scores do not reproduce AUC: ' + name + '/' + group)
        result[name] = {'videos': len(videos), 'score_columns': len(report['models']), 'auc_reproduced': True}
    p.dump(output/'AUC_REPRODUCTION.json', result)


def training_and_selection(root, config, rows, caches, anchor, ident):
    reference = reference_model(config)
    reference_folder = root/'reference'
    reference_folder.mkdir(exist_ok=True)
    ref_data = source_data(reference, reference_folder, rows, caches, config)
    ref_score = auc(ref_data, probabilities(ref_data))
    anchor_score = auc(ref_data, probabilities(ref_data, 0))
    expected = read(p.ROOT/'gate_study_v3_20261027/source_val_METRICS.json')
    if abs(ref_score-expected['NoGate']['video_auc']) > 1e-10 or abs(anchor_score-expected['Semantic']['video_auc']) > 1e-10:
        raise ValueError('Strong frozen source baseline did not reproduce; new training forbidden')
    if not (reference_folder/'REPRODUCTION.json').exists():
        p.dump(reference_folder/'REPRODUCTION.json', {'time': p.stamp(), 'reference_auc': ref_score,
               'semantic_auc': anchor_score, 'reference_sha256': REFERENCE_SHA,
               'semantic_sha256': ANCHOR_SHA, 'frozen_predictions_auc_exact': True})
    joint.status(root, 'strong_reference_reproduced', status='running',
                 old_fixed_source_val_auc=ref_score, old_semantic_source_val_auc=anchor_score)
    if (root/'PROMOTION_LOCK.json').exists():
        del reference
        release()
        return read(root/'PROMOTION_LOCK.json')
    screening = {}
    for name, extra in ARCHITECTURES.items():
        folder = root/'screening_seed2027'/name
        model = trained_model(SEEDS[0], config, {**config['model'], **extra}, anchor,
                              rows, caches, folder, ident, root)
        data = source_data(model, folder, rows, caches, config)
        screening[name] = {'folder': str(folder), 'source_val_fixed_auc': auc(data, probabilities(data))}
        if name == 'DW':
            # The same seed, base architecture, data, optimizer, sampling and budget.
            # Tiny hardware/rounding effects are tolerated, large bridge changes stop.
            delta = screening[name]['source_val_fixed_auc']-ref_score
            if not (folder/'BRIDGE_REPRODUCTION.json').exists():
                p.dump(folder/'BRIDGE_REPRODUCTION.json', {'difference_from_old_v2': delta,
                       'tolerance_auc': .0001, 'pass': abs(delta) <= .0001,
                       'scope': 'correction retraining reproduction; old frozen AUC separately exact'})
            if abs(delta) > .0001:
                raise ValueError('DW bridge differs from old V2 by more than 0.01 pp; investigate before comparisons')
        del model
        release()
    nominee = max(screening, key=lambda k: (screening[k]['source_val_fixed_auc'], k == 'DW'))
    if not (root/'SCREENING.json').exists():
        p.dump(root/'SCREENING.json', {'nominee': nominee, 'candidates': screening,
               'rule': 'source VAL g=1 only; no gate or target scores used'})
    source = {}
    nominee_folders = {}
    for seed in SEEDS:
        if seed == SEEDS[0]:
            folder = Path(screening[nominee]['folder'])
        else:
            folder = root/('seed'+str(seed))/nominee
        model = trained_model(seed, config, {**config['model'], **ARCHITECTURES[nominee]},
                              anchor, rows, caches, folder, ident, root)
        source[seed] = source_data(model, folder, rows, caches, config)
        nominee_folders[str(seed)] = str(folder)
        del model
        if seed != SEEDS[0] and nominee != 'DW':
            bridge = root/('seed'+str(seed))/'DW_Bridge'
            model = trained_model(seed, config, {**config['model'], **ARCHITECTURES['DW']},
                                  anchor, rows, caches, bridge, ident, root)
            source_data(model, bridge, rows, caches, config)
            del model
        release()
    fixed_scores = [probabilities(source[s]) for s in SEEDS]
    reference_scores = [probabilities(ref_data)] * len(SEEDS)
    architecture_evidence = paired(ref_data, fixed_scores, reference_scores)
    accepted = architecture_evidence['passes_registered_rule']
    chosen_architecture = nominee if accepted else 'OldV2'
    chosen_folders = nominee_folders if accepted else {str(s): str(REFERENCE) for s in SEEDS}
    if not (root/'ARCHITECTURE_SELECTION.json').exists():
        p.dump(root/'ARCHITECTURE_SELECTION.json', {'nominee': nominee, 'chosen': chosen_architecture,
               'new_architecture_promoted': accepted, 'evidence': architecture_evidence,
               'source_only': True, 'reference_retained_if_not_promoted': True})
    joint.status(root, 'architecture_confirmed', status='running', nominee=nominee,
                 chosen=chosen_architecture, evidence=architecture_evidence)

    # Gate optimization is conditional on the already-fixed expert choice.
    chosen_data = source if accepted else {s: ref_data for s in SEEDS}
    learned = {}
    for seed in SEEDS:
        folder = Path(chosen_folders[str(seed)])
        gate_folder = root/'gates'/('seed'+str(seed) if accepted else 'OldV2')
        gate_folder.mkdir(parents=True, exist_ok=True)
        model = joint.load_model(folder, anchor) if accepted else reference
        gate_fit = source_data(model, gate_folder, rows, caches, config, 'gate_fit')
        gate_holdout = source_data(model, gate_folder, rows, caches, config, 'gate_holdout')
        learned[seed] = fit_bounded(gate_fit, gate_holdout, gate_folder)
        if accepted:
            del model
    baseline = [probabilities(chosen_data[s]) for s in SEEDS]
    gate_evidence = {}
    for name in [*SCALES, 'Bounded']:
        scores = [(routing.video_probabilities(chosen_data[s], gate_values(chosen_data[s], learned[s]))
                   if name == 'Bounded' else probabilities(chosen_data[s], SCALES[name])) for s in SEEDS]
        gate_evidence[name] = paired(ref_data, scores, baseline) if name != 'NoGate' else {
            'mean_difference': 0., 'by_correction_seed': [0., 0., 0.],
            'family_cluster_95_ci': [0., 0.], 'passes_registered_rule': False}
    best_gate = max(gate_evidence, key=lambda k: (gate_evidence[k]['mean_difference'], k == 'NoGate'))
    chosen_gate = best_gate if gate_evidence[best_gate]['passes_registered_rule'] else 'NoGate'
    gate_paths = {str(s): str(root/'gates'/('seed'+str(s) if accepted else 'OldV2')/'Bounded.pt') for s in SEEDS}
    p.dump(root/'GATE_SELECTION.json', {'chosen': chosen_gate, 'raw_best': best_gate,
           'evidence': gate_evidence, 'gate_fitted_on': 'gate_fit', 'checkpoint_selected_on': 'gate_holdout',
           'family_selected_on': 'source_val', 'source_only': True})

    groups = {}
    for seed in SEEDS:
        groups[str(seed)] = {'Full': chosen_folders[str(seed)]}
        if accepted:
            ablations = {k: v for k, v in joint.ABLATIONS.items() if k not in ['Full', 'NoResidual']}
            if ARCHITECTURES[nominee]['injection'] != 'baseline':
                ablations['NoInjectionScale'] = {'injection': 'baseline'}
            base = {**config['model'], **ARCHITECTURES[nominee]}
            for name, diff in ablations.items():
                spec = {**base, **diff}
                if sum(base[k] != spec[k] for k in base) != 1:
                    raise ValueError('Ablation must change exactly one field')
                folder = root/('seed'+str(seed))/'ablations'/name
                model = trained_model(seed, config, spec, anchor, rows, caches, folder, ident, root)
                groups[str(seed)][name] = str(folder)
                if chosen_gate == 'Bounded':
                    gf = folder/'bounded_gate'
                    gf.mkdir(exist_ok=True)
                    fit = source_data(model, gf, rows, caches, config, 'gate_fit')
                    holdout = source_data(model, gf, rows, caches, config, 'gate_holdout')
                    fit_bounded(fit, holdout, gf)
                del model
                release()
    paths = {str(path): p.digest(path) for path in sorted(root.rglob('*.pt'))}
    paths[str(REFERENCE/'best.pt')] = REFERENCE_SHA
    promotion = {'time': p.stamp(), 'architecture': chosen_architecture, 'gate': chosen_gate,
                 'new_architecture_promoted': accepted, 'gate_promoted': chosen_gate != 'NoGate',
                 'groups': groups, 'gate_paths': gate_paths, 'screening': screening,
                 'nominee_folders': nominee_folders, 'checkpoint_sha256': paths,
                 'source_only_selection': True, 'target_scores_read_by_selector': False,
                 'shared_historical_semantic_anchor': True, 'correction_seeds': SEEDS,
                 'ablations_run_only_for_promoted_new_architecture': accepted,
                 'scope': 'conditional-transfer optimization; exploratory source and targets'}
    p.dump(root/'PROMOTION_LOCK.json', promotion)
    del reference
    release()
    return promotion


@torch.no_grad()
def evaluate(root, config, promotion, ident):
    output = root/'evaluation'
    output.mkdir(exist_ok=True)
    if (output/'COMPLETE.json').exists():
        return read(output/'ALL_METRICS.json')
    for path, expected in promotion['checkpoint_sha256'].items():
        if p.digest(path) != expected:
            raise ValueError('Checkpoint changed after source-only selection: ' + path)
    anchor = compact.CompactSemantic(config['semantic_checkpoint']).cuda()
    folders = sorted({str(path.parent) for path in root.rglob('best.pt')} | {str(REFERENCE)})
    models, tags = {}, {}
    for folder in folders:
        path = Path(folder)
        tag = 'OldV2' if path == REFERENCE else str(path.relative_to(root)).replace('/', '_')
        models[folder] = reference_model(config) if path == REFERENCE else joint.load_model(path, anchor)
        tags[folder] = tag
    encoder = compact.CompactEncoder(config)
    learned = {str(s): load_bounded(promotion['gate_paths'][str(s)]) for s in SEEDS}
    ablation_gates = {}
    if promotion['gate'] == 'Bounded':
        for seed, groups in promotion['groups'].items():
            for name, folder in groups.items():
                if name != 'Full':
                    ablation_gates[(seed, name)] = load_bounded(Path(folder)/'bounded_gate/Bounded.pt')
    manifests = p.ROOT/'guarded_dual_20261025/run_v1/manifests'
    names = ['source_val', 'source_test', *config['targets']]
    lock = {'time': p.stamp(), 'identity': ident, 'promotion_sha256': p.digest(root/'PROMOTION_LOCK.json'),
            'checkpoint_sha256': promotion['checkpoint_sha256'],
            'manifest_sha256': {n: p.digest(manifests/(n+'.json')) for n in names},
            'no_target_selection': True, 'model_registry': tags}
    if (output/'EVAL_LOCK.json').exists():
        previous = read(output/'EVAL_LOCK.json')
        if any(previous[k] != lock[k] for k in lock if k != 'time'):
            raise ValueError('Evaluation lock changed on resume')
    else:
        p.dump(output/'EVAL_LOCK.json', lock)
    reports = {}
    for name in names:
        result_path = output/(name+'_METRICS.json')
        if result_path.exists():
            reports[name] = read(result_path)
            continue
        if (output/(name+'_VIDEOS.jsonl')).exists():
            raise RuntimeError('Partial evaluation preserved; inspect before repair: ' + name)
        joint.status(root, 'frozen_evaluation', status='running', dataset=name)
        rows = read(manifests/(name+'.json'))
        ix = p.groups(rows)
        if any(len(i) != 32 for i in ix):
            raise ValueError('Every evaluation video must have exactly32 frames')
        predictions = {folder: {k: [] for k in ['s', 'd', 'r']} for folder in folders}
        loader = DataLoader(p.Pixels(rows), batch_size=config['batch_size'],
                            num_workers=config['workers'], pin_memory=True)
        done = 0
        for rgb in loader:
            tokens = compact.compact_encode(encoder, rgb)
            pixels = rgb.cuda(non_blocking=True)
            for folder, model in models.items():
                out = model(pixels, tokens)
                if model.spec['use_semantic']:
                    torch.testing.assert_close(out['semantic_logit'], tokens[:, 0, 0], rtol=0, atol=0)
                for key, field in [('s', 'semantic_logit'), ('d', 'delta'), ('r', 'residual_logit')]:
                    values = out.get(field, torch.zeros_like(out['logit']))
                    if not torch.isfinite(values).all():
                        raise ValueError('Nonfinite evaluation outputs')
                    predictions[folder][key].extend(values.cpu().tolist())
            done += len(rgb)
            if done % 2048 < len(rgb) or done == len(rows):
                p.log('V5_EVAL_PROGRESS', dataset=name, frames=done, total=len(rows))
        if done != len(rows):
            raise ValueError('Incomplete frame coverage')
        data = {folder: video_data(rows, pred) for folder, pred in predictions.items()}
        reference = data[str(REFERENCE)]
        scores = {'SemanticAnchor': probabilities(reference, 0),
                  'OldV2_Fixed': probabilities(reference)}
        for folder in folders:
            scores[tags[folder]+'_Fixed'] = probabilities(data[folder])
        for seed in SEEDS:
            key = str(seed)
            chosen = data[promotion['groups'][key]['Full']]
            scores['FullFixed_seed'+key] = probabilities(chosen)
            scores['FullBounded_seed'+key] = routing.video_probabilities(chosen, gate_values(chosen, learned[key]))
            for gate_name, scale in SCALES.items():
                scores[gate_name+'_seed'+key] = probabilities(chosen, scale)
            scores['Full_seed'+key] = (scores['FullBounded_seed'+key] if promotion['gate'] == 'Bounded'
                                        else scores[promotion['gate']+'_seed'+key])
            for group, folder in promotion['groups'][key].items():
                if group == 'Full':
                    continue
                one = data[folder]
                if promotion['gate'] == 'Bounded':
                    value = routing.video_probabilities(one, gate_values(one, ablation_gates[(key, group)]))
                else:
                    value = probabilities(one, SCALES[promotion['gate']])
                scores[group+'_seed'+key] = value
        label = reference['y'].cpu().numpy()
        arrays = {k: v.cpu().numpy() for k, v in scores.items()}
        report = {'videos': len(ix), 'frames': len(rows),
                  'real_videos': int((label == 0).sum()), 'fake_videos': int((label == 1).sum()),
                  'manifest_sha256': lock['manifest_sha256'][name],
                  'models': {k: {'video_auc': float(roc_auc_score(label, v))} for k, v in arrays.items()}}
        report['summary'] = {}
        group_names = list(promotion['groups'][str(SEEDS[0])]) + ['FullFixed', 'FullBounded']
        for group in group_names:
            values = [report['models'][group+'_seed'+str(seed)]['video_auc'] for seed in SEEDS]
            report['summary'][group] = {'mean_video_auc': float(np.mean(values)),
                                         'std_video_auc': float(np.std(values, ddof=1)),
                                         'scope': 'three correction seeds sharing one frozen semantic expert'}
        with (output/(name+'_VIDEOS.jsonl')).open('x', encoding='utf-8') as f:
            for i, video in enumerate(reference['video_id']):
                f.write(json.dumps({'video_id': video, 'family_id': reference['family_id'][i],
                        'label': int(label[i]), 'scores': {k: float(v[i]) for k, v in arrays.items()}}, allow_nan=False)+'\n')
        joint.status(root, 'paired_statistics', status='running', dataset=name)
        report['paired'] = {}
        full = [scores['Full_seed'+str(seed)] for seed in SEEDS]
        for control in ['SemanticAnchor', 'OldV2_Fixed', 'FullFixed']:
            ref = ([scores['FullFixed_seed'+str(seed)] for seed in SEEDS] if control == 'FullFixed'
                   else [scores[control]] * len(SEEDS))
            report['paired'][control] = paired(reference, full, ref)
        p.dump(result_path, report)
        reports[name] = report
        p.log('V5_DATASET_COMPLETE', dataset=name, summary=report['summary'],
              old_fixed_auc=report['models']['OldV2_Fixed']['video_auc'])
        del predictions, data, scores, arrays
        release()
    if identity(config) != ident:
        raise ValueError('Code/backbone/reference changed during experiment')
    for path, expected in promotion['checkpoint_sha256'].items():
        if p.digest(path) != expected:
            raise ValueError('Frozen checkpoint changed during evaluation')
    verify_saved_auc(output, names)
    p.dump(output/'ALL_METRICS.json', reports)
    p.dump(output/'COMPLETE.json', {'time': p.stamp(), 'datasets': names,
           'no_target_model_selection': True, 'shared_semantic_anchor': True})
    del models, encoder, anchor
    release()
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    lease = p.acquire_gpu()
    config = configuration()
    root = Path(args.output).resolve()
    ident = identity(config)
    if root.exists():
        if not args.resume or read(root/'IDENTITY.json') != ident or read(root/'CONFIG.json') != config:
            raise ValueError('New output directory required, or exact completed-stage resume')
    else:
        root.mkdir(parents=True)
        p.dump(root/'CONFIG.json', config)
        p.dump(root/'IDENTITY.json', ident)
        shutil.copyfile(__file__, root/'EXECUTED_SOURCE.py')
    try:
        joint.status(root, 'gpu_preflight', status='running')
        if not (root/'GPU_PREFLIGHT.json').exists():
            gpu_preflight(config, root)
        if args.audit_only:
            joint.status(root, 'gpu_preflight_completed', status='completed', performance_evidence=False)
            return
        if shutil.disk_usage(root).free < 1024**3:
            raise RuntimeError('Less than1GiB disk free; no historical files removed')
        if (root/'PROMOTION_LOCK.json').exists():
            promotion = read(root/'PROMOTION_LOCK.json')
        else:
            joint.status(root, 'source_manifests_and_memory_preflight', status='running')
            rows = p.manifests(config)
            compact.memory_preflight(config)
            if not (root/'DATA_SELECTION.json').exists():
                p.dump(root/'DATA_SELECTION.json', {name: {'frames': len(part), 'videos': len(p.groups(part)),
                       'families': len({r['family_id'] for r in part})} for name, part in rows.items()})
            joint.status(root, 'strong_anchor_compact_cache', status='running')
            encoder = compact.CompactEncoder(config)
            caches = {name: compact.compact_cache(encoder, part, config) for name, part in rows.items()}
            caches['fit_aug'] = compact.compact_cache(encoder, rows['fit'], config, augmented=True)
            del encoder
            release()
            anchor = compact.CompactSemantic(config['semantic_checkpoint']).cuda()
            promotion = training_and_selection(root, config, rows, caches, anchor, ident)
            del anchor, caches
            release()
        reports = evaluate(root, config, promotion, ident)
        p.dump(root/'FINAL_REPORT.json', {'time': p.stamp(),
               'architecture': promotion['architecture'], 'gate': promotion['gate'],
               'new_architecture_promoted': promotion['new_architecture_promoted'],
               'gate_promoted': promotion['gate_promoted'],
               'full_auc': {name: r['summary']['Full'] for name, r in reports.items()},
               'old_fixed_auc': {name: r['models']['OldV2_Fixed']['video_auc'] for name, r in reports.items()},
               'source_goal_met': reports['source_val']['summary']['Full']['mean_video_auc'] >= .95,
               'each_target_goal_met': {name: reports[name]['summary']['Full']['mean_video_auc'] >= .85 for name in config['targets']},
               'targets_exploratory': True, 'shared_historical_semantic_anchor': True,
               'no_targetwise_model_choice': True, 'ablation_rankings_preserved': True})
        joint.status(root, 'completed', status='completed')
    except Exception as error:
        joint.status(root, 'failed', status='failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
