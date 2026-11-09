"""Remote CUDA joint residual/router study with fresh, family-isolated experts.

Historical scripts/checkpoints are immutable dependencies. No target selection.
"""
import argparse
import collections
import copy
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

import gpu_pipeline as p
import full_source_v2 as compact
import gate_study_v3 as routing
from model import initialization


ARCHITECTURES = {
    'DW': {'encoder_kind': 'dw', 'injection': 'baseline', 'semantic_shortcut': True},
    'DW_LayerScale': {'encoder_kind': 'dw', 'injection': 'layerscale', 'semantic_shortcut': True},
    'DW_ReZero': {'encoder_kind': 'dw', 'injection': 'rezero', 'semantic_shortcut': True},
    'IR_LayerScale': {'encoder_kind': 'inverted', 'injection': 'layerscale', 'semantic_shortcut': True},
    'ResidualRequired_LayerScale': {'encoder_kind': 'dw', 'injection': 'layerscale', 'semantic_shortcut': False},
}
CONSTANTS = {'Fixed025': .25, 'Fixed050': .5, 'Fixed075': .75, 'NoGate': 1.}
ABLATIONS = {'Full': {}, 'NoResidual': {'use_residual': False},
             'NoSemantic': {'use_semantic': False}, 'NoFilter': {'use_filter': False},
             'NoInteraction': {'use_interaction': False}, 'NoAdaptation': {'adapt_scale': 0.}}
SEEDS = [2027, 3407, 4099]
RAW_ENCODE = p.encode


def release():
    gc.collect()
    torch.cuda.empty_cache()


def status(root, stage, **values):
    record = {'stage': stage, 'time': p.stamp(), **values}
    # Historical dump() deliberately uses exclusive creation; live state must replace.
    temporary = root / '.STATE.pending.json'
    with temporary.open('w', encoding='utf-8') as file:
        json.dump(record, file, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, root / 'STATE.json')
    with (root / 'STAGES.jsonl').open('a', encoding='utf-8') as file:
        file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    p.log('V4_STATE', **record)


class FreshSemantic(nn.Module):
    """Same full token architecture as the historical semantic expert, fresh head."""
    def __init__(self):
        super().__init__()
        self.semantic_query = nn.Linear(1024, 192)
        self.fusion = p.GatedResidualCrossAttention(192, 1024, 256, 8, 128, .1,
                                                  use_gate=True, use_residual_pool=True)
        self.classifier = p.BinaryClassifierHead(128, 64, .1)

    def forward(self, tokens):
        return self.classifier(self.fusion(self.semantic_query(tokens.float()), tokens.float()))


class SemanticExit(nn.Module):
    def __init__(self, semantic):
        super().__init__()
        self.expert = copy.deepcopy(semantic).requires_grad_(False).eval()

    def train(self, mode=True):
        return super().train(False)

    def forward(self, tokens):
        return tokens[:, 0, 0]


class InvertedBlock(nn.Module):
    def __init__(self, channels, expansion=2):
        super().__init__()
        expanded = channels * expansion
        self.block = nn.Sequential(
            nn.Conv2d(channels, expanded, 1, bias=False), nn.GroupNorm(8, expanded), nn.GELU(),
            nn.Conv2d(expanded, expanded, 3, padding=1, groups=expanded, bias=False),
            nn.GroupNorm(8, expanded), nn.GELU(),
            nn.Conv2d(expanded, channels, 1, bias=False), nn.GroupNorm(8, channels))

    def forward(self, x):
        return x + self.block(x)


class JointDual(compact.CompactDual):
    def __init__(self, semantic, spec):
        super().__init__(semantic, spec)
        h = spec['hidden']
        if spec['encoder_kind'] == 'inverted':
            # Keep the same output width and grid, change only the encoder family.
            with initialization('v4_inverted_encoder'):
                self.encoder = nn.Sequential(
                    nn.Conv2d(32, h, 1, bias=False), nn.GroupNorm(8, h), nn.GELU(),
                    nn.AvgPool2d(2), InvertedBlock(h), nn.AvgPool2d(2), InvertedBlock(h))
        if spec['injection'] == 'layerscale':
            # Scale the h hidden correction channels, after GELU, before the zero-init head.
            self.channel_scale = nn.Parameter(torch.full((h,), .01))
        elif spec['injection'] == 'rezero':
            self.alpha = nn.Parameter(torch.zeros(()))
            # Avoid the double-zero dead gradient: alpha=0 requires a nonzero last layer.
            with initialization('v4_rezero_last'):
                nn.init.normal_(self.correction[-1].weight, std=.001)
                nn.init.zeros_(self.correction[-1].bias)

    def forward(self, rgb, tokens):
        b = len(rgb)
        has_s, has_r = self.spec['use_semantic'], self.spec['use_residual']
        if not has_s and not has_r:
            raise ValueError('Both branches disabled')
        z_s = self.semantic(tokens) if has_s else rgb.new_zeros(b)
        if not has_r:
            return {'logit': z_s, 'semantic_logit': z_s, 'delta': torch.zeros_like(z_s)}
        grid = math.isqrt(tokens.shape[1] - 1)
        if grid * grid != tokens.shape[1] - 1:
            raise ValueError('Non-square tokens')
        r = F.adaptive_avg_pool2d(self.encoder(self.front(rgb)), (grid, grid)).flatten(2).transpose(1, 2)
        z_r = self.r_head(r.mean(1)).squeeze(-1)
        s = self.s_project(tokens[:, 1:].float()) if has_s else torch.zeros_like(r)
        u = torch.zeros_like(r)
        if self.spec['use_interaction'] and has_s:
            yy, xx = torch.meshgrid(torch.arange(grid, device=r.device),
                                    torch.arange(grid, device=r.device), indexing='ij')
            coords = torch.stack([xx, yy], -1).reshape(-1, 2)
            mask = (coords[:, None] - coords[None, :]).abs().amax(-1) > self.spec['radius']
            memory = self.kv_norm(r)
            u, _ = self.cross(self.q_norm(s), memory, memory, attn_mask=mask, need_weights=False)
        direct_s = s.mean(1) if self.spec['semantic_shortcut'] else torch.zeros_like(s.mean(1))
        hidden = self.correction[:-1](torch.cat([direct_s, r.mean(1), u.mean(1)], -1))
        if self.spec['injection'] == 'layerscale':
            hidden = hidden * self.channel_scale
        raw = self.correction[-1](hidden).squeeze(-1)
        delta = self.spec['correction_bound'] * torch.tanh(raw / self.spec['correction_bound'])
        if self.spec['injection'] == 'rezero':
            delta = self.alpha.tanh() * delta
        return {'logit': z_s + delta, 'semantic_logit': z_s, 'delta': delta, 'residual_logit': z_r}


def config_v4():
    config = p.configuration()
    config.update(name='joint_residual_gate_v4_fresh_20261027', seeds=SEEDS,
                  epochs=10, semantic_epochs=10, semantic_control_epochs=20,
                  semantic_lr=.0003, batch_size=16, workers=2,
                  fit_families=10**9, pilot_val_families=10**9, gate_families_per_fold=10**9,
                  gate_steps=600, architectures=ARCHITECTURES, ablations=ABLATIONS,
                  source_only_selection=True, target_scope='exploratory follow-up; prior target scores already observed',
                  semantic_epoch_validation_frames=8, final_evaluation_frames=32,
                  semantic_initialization='pretrained frozen CLIP; all semantic_query/fusion/classifier parameters fresh',
                  family_isolation='neither fresh semantic nor residual training sees gate_fit/gate_holdout families',
                  selection='seed2027 screens architectures+routers on source_val32; freeze architecture/router family for all seeds',
                  semantic_budget_control='20 semantic epochs vs 10 semantic+10 residual; optimizer-step matched, not FLOP matched',
                  performance_goals={'source_val_auc': .95, 'each_target_auc': .85})
    return config


def identity(config):
    result = p.identities(config)
    for module in [sys.modules[__name__], compact, routing]:
        path = Path(module.__file__).resolve()
        result[str(path)] = p.digest(path)
    return result


def sampler_weights(rows):
    counts = collections.Counter(r['method'] for r in rows)
    fake_methods = {r['method'] for r in rows if r['label']}
    return [(.5 if not r['label'] else .5 / len(fake_methods)) / counts[r['method']] for r in rows]


@torch.no_grad()
def semantic_predictions(model, cache, batch):
    model.eval()
    values = []
    for start in range(0, len(cache['tokens']), batch):
        values.extend(model(cache['tokens'][start:start+batch].cuda().float()).sigmoid().cpu().tolist())
    return np.asarray(values, np.float32)


def semantic_train(seed, config, rows, caches, folder, ident):
    folder.mkdir(exist_ok=False)
    p.seed_all(seed)
    model = FreshSemantic().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['semantic_lr'], weight_decay=config['weight_decay'])
    y = torch.tensor([r['label'] for r in rows['fit']], device='cuda', dtype=torch.float32)
    weights = sampler_weights(rows['fit'])
    best_main, best_control = -math.inf, -math.inf
    for epoch in range(config['semantic_control_epochs'] + 1):
        started = time.monotonic()
        total, updates = 0., 0
        if epoch:
            model.train()
            generator = torch.Generator().manual_seed(seed + 10000 * epoch)
            indexes = list(WeightedRandomSampler(weights, len(y), True, generator=generator))
            for start in range(0, len(indexes), config['batch_size']):
                ids = indexes[start:start+config['batch_size']]
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for view in ['fit', 'fit_aug']:
                    logits = model(caches[view]['tokens'][ids].cuda().float())
                    losses.append(F.binary_cross_entropy_with_logits(logits, y[ids]))
                loss = torch.stack(losses).mean()
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite semantic loss')
                loss.backward()
                norm = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach()); updates += 1
        probability = semantic_predictions(model, caches['source_val'], config['batch_size'])
        score = p.metrics(rows['source_val'], probability)
        record = {'seed': seed, 'epoch': epoch, **score, 'loss': total / max(1, updates),
                  'updates': updates, 'seconds': time.monotonic() - started,
                  'scope': '8 deterministic validation frames/video; final reports use 32'}
        with (folder / 'history.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        state = {'state': model.state_dict(), 'seed': seed, 'epoch': epoch, 'score': score, 'identity': ident}
        if epoch <= config['semantic_epochs'] and score['video_auc'] > best_main:
            best_main = score['video_auc']; torch.save(state, folder / 'main.pt')
        if score['video_auc'] > best_control:
            best_control = score['video_auc']; torch.save(state, folder / 'budget_control.pt')
        p.log('FRESH_SEMANTIC_EPOCH', **record)
    p.dump(folder / 'COMPLETE.json', {'seed': seed, 'main_auc8': best_main, 'budget_control_auc8': best_control,
           'epochs_run': config['semantic_control_epochs'], 'identity': ident})
    del model, optimizer
    release()


def load_semantic(folder, control=False):
    state = torch.load(folder / ('budget_control.pt' if control else 'main.pt'), map_location='cpu', weights_only=True)
    model = FreshSemantic().cuda()
    model.load_state_dict(state['state'], strict=True)
    return model.eval().requires_grad_(False)


class FreshCompactEncoder(nn.Module):
    def __init__(self, clip, semantic):
        super().__init__()
        self.clip, self.anchor = clip, semantic


@torch.no_grad()
def fresh_encode(encoder, rgb):
    raw = RAW_ENCODE(encoder.clip, rgb)
    result = raw.new_zeros((len(raw), 257, 192))
    result[:, 1:] = encoder.anchor.semantic_query(raw[:, 1:]).half().float()
    result[:, 0, 0] = encoder.anchor(raw)
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite fresh compact features')
    return result


def fit_residual(seed, config, spec, semantic, rows, caches, folder, ident):
    folder.mkdir(exist_ok=False)
    p.seed_all(seed)
    model = JointDual(SemanticExit(semantic), spec).cuda()
    parameters = [v for v in model.parameters() if v.requires_grad]
    filters = [model.front.delta] if model.front.delta.requires_grad else []
    filter_ids = {id(v) for v in filters}
    optimizer = torch.optim.AdamW([
        {'params': [v for v in parameters if id(v) not in filter_ids], 'lr': config['dual_lr']},
        {'params': filters, 'lr': config['dual_lr'] * config['filter_lr_multiplier']}], weight_decay=config['weight_decay'])
    labels = torch.tensor([r['label'] for r in rows['fit']], device='cuda', dtype=torch.float32)
    weights = sampler_weights(rows['fit'])
    best = -math.inf
    for epoch in range(config['epochs'] + 1):
        started = time.monotonic(); total = 0.; updates = 0
        if epoch and spec['use_residual']:
            model.train()
            generator = torch.Generator().manual_seed(seed + 10000 * epoch)
            indices = list(WeightedRandomSampler(weights, len(labels), True, generator=generator))
            for start in range(0, len(indices), config['batch_size']):
                ids = indices[start:start+config['batch_size']]
                optimizer.zero_grad(set_to_none=True)
                outputs, losses = [], []
                for view in ['fit', 'fit_aug']:
                    result = model(caches[view]['rgb'][ids].cuda().float()/255,
                                   caches[view]['tokens'][ids].cuda().float())
                    losses.append(F.binary_cross_entropy_with_logits(result['logit'], labels[ids]) +
                                  config['residual_aux_weight'] * F.binary_cross_entropy_with_logits(result['residual_logit'], labels[ids]))
                    outputs.append(result)
                loss = torch.stack(losses).mean() + config['consistency_weight'] * F.mse_loss(outputs[0]['delta'], outputs[1]['delta'])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite residual objective')
                loss.backward()
                if epoch == 1 and updates == 0:
                    encoder_gradient = any(v.grad is not None and bool(v.grad.abs().sum()) for v in model.encoder.parameters())
                    injection_gradient = (bool(model.alpha.grad.abs().sum()) if spec['injection'] == 'rezero'
                                          else bool(model.correction[-1].weight.grad.abs().sum()))
                    if not encoder_gradient or not injection_gradient:
                        raise RuntimeError('Residual auxiliary or injection has a dead first-step gradient')
                    p.dump(folder/'GRADIENT_CONTRACT.json', {'encoder_auxiliary_gradient_nonzero': encoder_gradient,
                           'injection_gradient_nonzero': injection_gradient, 'injection': spec['injection'],
                           'device': torch.cuda.get_device_name()})
                nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step(); updates += 1; total += float(loss.detach())
        pred = p.predict(model, caches['source_val'], config['batch_size'])
        score = p.metrics(rows['source_val'], p.probability(pred))
        record = {'seed': seed, 'group': folder.name, 'epoch': epoch, **score,
                  'loss': total/max(1, updates), 'updates': updates, 'seconds': time.monotonic()-started}
        with (folder/'history.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        if score['video_auc'] > best:
            best = score['video_auc']
            torch.save({'model': model.state_dict(), 'spec': spec, 'seed': seed, 'epoch': epoch,
                        'score': score, 'identity': ident}, folder/'best.pt')
        p.log('RESIDUAL_EPOCH', **record)
        if not spec['use_residual']:
            break
    saved = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(saved['model'], strict=True); model.eval()
    p.dump(folder/'SELECTION.json', {'epoch': saved['epoch'], **saved['score'], 'spec': spec,
                                   'rule': 'source_val32 video AUC, epoch0 eligible'})
    return model


def data_from_predictions(rows, pred):
    pred = {k: np.asarray(values, np.float32) for k, values in pred.items()}
    s, d, r, y, ix = p.video_tensors(rows, pred)
    return {'s': s, 'd': d, 'r': r, 'y': y,
            'video_id': [rows[i[0]]['video_id'] for i in ix],
            'family_id': [rows[i[0]].get('family_id', rows[i[0]]['video_id']) for i in ix]}


@torch.no_grad()
def selected_gate(name, data, fitted):
    if name in CONSTANTS:
        return torch.full_like(data['y'], CONSTANTS[name])
    return routing.gate_for(name, data, fitted)


def gate_study(config, model, rows, caches, folder, chosen_family=None):
    out = folder/'gates'; out.mkdir(exist_ok=False)
    datasets = {}
    for name in ['gate_fit', 'gate_holdout', 'source_val']:
        pred = p.predict(model, caches[name], config['batch_size'])
        p.save_scores(out/(name+'_frames.jsonl'), rows[name], pred)
        datasets[name] = data_from_predictions(rows[name], pred)
    families = list(routing.SPECS) if chosen_family is None else ([chosen_family] if chosen_family not in CONSTANTS else [])
    fitted = {name: routing.fit_one(name, routing.SPECS[name], datasets['gate_fit'], datasets['gate_holdout'], out)
              for name in families}
    names = [*CONSTANTS, *families]
    report = {}
    for part, data in datasets.items():
        report[part] = {name: routing.measure(data, selected_gate(name, data, fitted))[0] for name in names}
        report[part]['Semantic'] = routing.measure(data, torch.zeros_like(data['y']))[0]
    winner = chosen_family or max(names, key=lambda name: (report['source_val'][name]['video_auc'], name == 'NoGate',
                                                         -report['source_val'][name]['balanced_video_bce']))
    selection = {'winner': winner, 'source_val': report['source_val'][winner], 'source_only': True,
                 'rule': 'router checkpoint by gate_holdout; family by source_val in screening seed; family fixed in confirmation seeds',
                 'semantic_anchor_auc': report['source_val']['Semantic']['video_auc'],
                 'fusion_gain_over_semantic': report['source_val'][winner]['video_auc']-report['source_val']['Semantic']['video_auc']}
    p.dump(out/'ALL_METRICS.json', report); p.dump(out/'SELECTION.json', selection)
    p.dump(out/'COMPLETE.json', {'time': p.stamp(), 'expert_sha256': p.digest(folder/'best.pt')})
    p.log('V4_GATE_SELECTION', group=str(folder), **selection)
    return selection


def load_model(folder, semantic):
    state = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)
    model = JointDual(SemanticExit(semantic), state['spec']).cuda()
    model.load_state_dict(state['model'], strict=True)
    return model.eval()


def load_gate(folder, name):
    if name in CONSTANTS:
        return {}
    info = torch.load(folder/'gates'/(name+'.pt'), map_location='cpu', weights_only=True)
    router = routing.Router(info['spec']['router']).cuda()
    router.load_state_dict(info['state'], strict=True)
    return {name: {'model': router.eval(), 'spec': info['spec'], 'center': info['center'].cuda(), 'scale': info['scale'].cuda()}}


@torch.no_grad()
def gpu_contract(semantic, caches, config, output):
    rgb = caches['source_val']['rgb'][:2].cuda().float()/255
    tokens = caches['source_val']['tokens'][:2].cuda().float()
    checks = {}
    for name, extra in ARCHITECTURES.items():
        p.seed_all(SEEDS[0])
        model = JointDual(SemanticExit(semantic), {**config['model'], **extra}).cuda().eval()
        torch.testing.assert_close(model(rgb, tokens)['logit'], tokens[:, 0, 0], rtol=0, atol=0)
        checks[name] = {'initial_exact_semantic_exit': True, 'trainable_parameters': sum(v.numel() for v in model.parameters() if v.requires_grad)}
        del model
    p.dump(output/'GPU_CONTRACT.json', {'device': torch.cuda.get_device_name(), 'architectures': checks,
                                      'scope': 'initial exact exit; optimization gradients checked on first training step'})


@torch.no_grad()
def source_diagnostic(model, cache, rows, config, output):
    # Label-blind donor permutation from another family. Source only.
    ids = p.groups(rows)
    donor = []
    for i, video in enumerate(ids):
        j = (i + 1) % len(ids)
        while rows[ids[j][0]]['family_id'] == rows[video[0]]['family_id']:
            j = (j + 1) % len(ids)
        donor.extend(zip(video[::8], ids[j][::8]))
    clean, swapped = [], []
    sample_rows = []
    for start in range(0, len(donor), config['batch_size']):
        pairs = donor[start:start+config['batch_size']]
        a, b = [x[0] for x in pairs], [x[1] for x in pairs]
        tokens = cache['tokens'][a].cuda().float()
        clean.extend(model(cache['rgb'][a].cuda().float()/255, tokens)['delta'].cpu().tolist())
        swapped.extend(model(cache['rgb'][b].cuda().float()/255, tokens)['delta'].cpu().tolist())
        sample_rows.extend(rows[i] for i in a)
    a, b = np.asarray(clean), np.asarray(swapped)
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None
    result = {'frames': len(a), 'delta_std': float(a.std()), 'rgb_swap_mean_abs_change': float(np.abs(a-b).mean()),
              'rgb_swap_correlation': corr, 'scope': 'input sensitivity, not proof of useful residual evidence'}
    rgb = cache['rgb'][:config['batch_size']].cuda().float()/255
    t = cache['tokens'][:config['batch_size']].cuda().float()
    for _ in range(5): model(rgb, t)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20): model(rgb, t)
    end.record(); torch.cuda.synchronize()
    result['correction_forward_ms_per_batch'] = start.elapsed_time(end)/20
    result['timing_scope'] = 'GPU correction forward on compact features, excludes CLIP and semantic expert'
    p.dump(output/'RESIDUAL_DIAGNOSTIC.json', result)


def all_training(root, config, rows, ident):
    semantic_root = root/'semantic'; semantic_root.mkdir(exist_ok=True)
    pending = [s for s in SEEDS if not (semantic_root/str(s)/'COMPLETE.json').exists()]
    if pending:
        status(root, 'fresh_semantic_cache', status='running', seeds=pending)
        reduced = {**rows, 'source_val': [rows['source_val'][i] for ids in p.groups(rows['source_val']) for i in ids[::4]]}
        counts = [len(rows['fit']), len(rows['fit']), len(reduced['source_val'])]
        estimate = sum(counts)*(257*1024*2+3*224*224)
        limit = None
        for filename in ['/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes']:
            file = Path(filename)
            if file.exists() and file.read_text().strip().isdigit():
                limit = min(limit or int(file.read_text()), int(file.read_text()))
        if limit is None:
            raise RuntimeError('Cannot determine container memory limit')
        if estimate+6*1024**3 > limit:
            raise RuntimeError('Raw semantic caches exceed cgroup RAM budget')
        p.log('V4_RAW_MEMORY_PREFLIGHT', cache_bytes=estimate, reserve_bytes=6*1024**3, limit_bytes=limit)
        clip = p.load_encoder(config)
        caches = {name: p.memory_cache(clip, reduced[name], config) for name in ['fit', 'source_val']}
        caches['fit_aug'] = p.memory_cache(clip, reduced['fit'], config, augmented=True)
        del clip; release()
        for seed in pending:
            status(root, 'fresh_semantic_training', status='running', seed=seed)
            semantic_train(seed, config, reduced, caches, semantic_root/str(seed), ident)
        del caches; release()
    for seed in SEEDS:
        seed_root = root/('seed'+str(seed)); seed_root.mkdir(exist_ok=True)
        if (seed_root/'COMPLETE.json').exists():
            continue
        status(root, 'fresh_compact_cache', status='running', seed=seed)
        semantic = load_semantic(semantic_root/str(seed))
        clip = p.load_encoder(config)
        encoder = FreshCompactEncoder(clip, semantic)
        original = compact.compact_encode
        compact.compact_encode = fresh_encode
        try:
            caches = {name: compact.compact_cache(encoder, selected, config) for name, selected in rows.items()}
            caches['fit_aug'] = compact.compact_cache(encoder, rows['fit'], config, augmented=True)
        finally:
            compact.compact_encode = original
        del encoder, clip; release()
        if not (seed_root/'GPU_CONTRACT.json').exists():
            gpu_contract(semantic, caches, config, seed_root)
        if seed == SEEDS[0] and not (root/'JOINT_SELECTION.json').exists():
            screening = seed_root/'screening'; screening.mkdir(exist_ok=True)
            selections = {}
            for name, arch in ARCHITECTURES.items():
                folder = screening/name
                if (folder/'gates/COMPLETE.json').exists():
                    selections[name] = json.loads((folder/'gates/SELECTION.json').read_text())
                    continue
                status(root, 'joint_source_screening', status='running', seed=seed, architecture=name)
                model = fit_residual(seed, config, {**config['model'], **arch}, semantic, rows, caches, folder, ident)
                source_diagnostic(model, caches['source_val'], rows['source_val'], config, folder)
                selections[name] = gate_study(config, model, rows, caches, folder)
                del model; release()
            winner = max(selections, key=lambda name: (selections[name]['source_val']['video_auc'], name=='DW',
                                                       -selections[name]['source_val']['balanced_video_bce']))
            p.dump(root/'JOINT_SELECTION.json', {'architecture': winner, 'router_family': selections[winner]['winner'],
                   'screening_seed': seed, 'all_candidates': selections, 'source_only': True,
                   'target_scores_read_by_selector': False, 'time': p.stamp(), 'spec': ARCHITECTURES[winner]})
        joint = json.loads((root/'JOINT_SELECTION.json').read_text())
        variants = dict(ABLATIONS)
        if joint['spec']['injection'] != 'baseline':
            variants['NoInjectionScale'] = {'injection': 'baseline'}
        group_folders = {}
        for name, diff in variants.items():
            base_spec = {**config['model'], **joint['spec']}
            spec = {**base_spec, **diff}
            changed = [k for k in base_spec if base_spec[k] != spec[k]]
            if len(changed) != (0 if name == 'Full' else 1):
                raise ValueError('Ablation changes more/less than one field')
            if seed == SEEDS[0] and name == 'Full':
                folder = seed_root/'screening'/joint['architecture']
            else:
                folder = seed_root/name
                if not (folder/'gates/COMPLETE.json').exists():
                    status(root, 'full_single_component_ablation', status='running', seed=seed, group=name)
                    model = fit_residual(seed, config, spec, semantic, rows, caches, folder, ident)
                    gate_study(config, model, rows, caches, folder, chosen_family=joint['router_family'])
                    if name == 'Full':
                        source_diagnostic(model, caches['source_val'], rows['source_val'], config, folder)
                    del model; release()
            group_folders[name] = str(folder)
        p.dump(seed_root/'GROUPS.json', group_folders)
        p.dump(seed_root/'COMPLETE.json', {'time': p.stamp(), 'groups': group_folders, 'identity': ident})
        del semantic, caches; release()


@torch.no_grad()
def evaluate_all(root, config, ident):
    # Freeze all seeds, all variants and the single shared router family before targets.
    output = root/'evaluation'; output.mkdir(exist_ok=True)
    paths = sorted([*root.glob('semantic/*/*.pt'), *root.glob('seed*/**/*.pt')])
    hashes = {str(path): p.digest(path) for path in paths}
    lock = {'time': p.stamp(), 'identity': ident, 'checkpoints': hashes,
            'joint_selection_sha256': p.digest(root/'JOINT_SELECTION.json'),
            'source_only_selection': True, 'no_per_target_model_choice': True}
    if (output/'EVAL_LOCK.json').exists():
        old = json.loads((output/'EVAL_LOCK.json').read_text())
        if old['checkpoints'] != hashes or old['identity'] != ident:
            raise ValueError('Frozen models changed on resume')
    else:
        p.dump(output/'EVAL_LOCK.json', lock)
    joint = json.loads((root/'JOINT_SELECTION.json').read_text())
    family = joint['router_family']
    experts, controls, models, gates = {}, {}, {}, {}
    for seed in SEEDS:
        experts[seed] = load_semantic(root/'semantic'/str(seed))
        controls[seed] = load_semantic(root/'semantic'/str(seed), control=True)
        folders = json.loads((root/('seed'+str(seed))/'GROUPS.json').read_text())
        models[seed] = {name: load_model(Path(path), experts[seed]) for name, path in folders.items()}
        gates[seed] = {name: load_gate(Path(path), family) for name, path in folders.items()}
    clip = p.load_encoder(config)
    source = p.ROOT/'guarded_dual_20261025/run_v1/manifests'
    datasets = ['source_val', 'source_test', *config['targets']]
    all_metrics = {}
    for dataset in datasets:
        report_path = output/(dataset+'_METRICS.json')
        if report_path.exists():
            all_metrics[dataset] = json.loads(report_path.read_text()); continue
        status(root, 'frozen_full_evaluation', status='running', dataset=dataset)
        rows = json.loads((source/(dataset+'.json')).read_text())
        ix = p.groups(rows)
        if any(len(i) != 32 for i in ix):
            raise ValueError('All evaluation videos require exactly32 frames')
        predictions = {seed: {name: {k: [] for k in ['s', 'd', 'r']} for name in models[seed]} for seed in SEEDS}
        budget = {seed: [] for seed in SEEDS}
        loader = DataLoader(p.Pixels(rows), batch_size=config['batch_size'], num_workers=config['workers'], pin_memory=True)
        done = 0; started = time.monotonic()
        for rgb in loader:
            raw = RAW_ENCODE(clip, rgb)
            x = rgb.cuda(non_blocking=True)
            for seed in SEEDS:
                semantic = experts[seed]
                tokens = raw.new_zeros((len(raw), 257, 192))
                tokens[:, 1:] = semantic.semantic_query(raw[:, 1:]).half().float()
                tokens[:, 0, 0] = semantic(raw)
                budget[seed].extend(controls[seed](raw).sigmoid().cpu().tolist())
                for name, model in models[seed].items():
                    result = model(x, tokens)
                    for key, field in [('s', 'semantic_logit'), ('d', 'delta'), ('r', 'residual_logit')]:
                        values = result.get(field, torch.zeros_like(result['logit']))
                        if not torch.isfinite(values).all():
                            raise ValueError('Nonfinite frozen evaluation outputs')
                        predictions[seed][name][key].extend(values.cpu().tolist())
            done += len(rgb)
            if done % 2048 < len(rgb) or done == len(rows):
                p.log('V4_EVAL_PROGRESS', dataset=dataset, frames=done, total=len(rows), seconds=time.monotonic()-started)
        if done != len(rows):
            raise ValueError('Incomplete evaluation coverage')
        report = {'frames': len(rows), 'videos': len(ix), 'seeds': {}, 'manifest_sha256': p.digest(source/(dataset+'.json'))}
        for seed in SEEDS:
            seed_report = {}
            video_scores = {}
            for name, pred in predictions[seed].items():
                data = data_from_predictions(rows, pred)
                selected = selected_gate(family, data, gates[seed][name])
                seed_report[name], scores = routing.measure(data, selected)
                video_scores[name] = scores.cpu().tolist()
                if name == 'Full':
                    seed_report['NoGate'], nogate = routing.measure(data, torch.ones_like(data['y']))
                    video_scores['NoGate'] = nogate.cpu().tolist()
                    seed_report['ResidualAux'] = {'video_auc': routing.auc(data['y'], data['r'].sigmoid().mean(-1)),
                                                  'scope': 'auxiliary residual-only score; same frozen Full encoder'}
            control = np.asarray(budget[seed], np.float32)
            seed_report['SemanticBudget20'] = p.metrics(rows, control)
            video_scores['SemanticBudget20'] = [float(control[i].mean()) for i in ix]
            report['seeds'][str(seed)] = seed_report
            with (output/(dataset+'_seed'+str(seed)+'_VIDEOS.jsonl')).open('x') as f:
                for i, ids in enumerate(ix):
                    f.write(json.dumps({'video_id': rows[ids[0]]['video_id'], 'family_id': rows[ids[0]].get('family_id'),
                            'label': rows[ids[0]]['label'], 'scores': {k: values[i] for k, values in video_scores.items()}}, allow_nan=False)+'\n')
        names = list(report['seeds'][str(SEEDS[0])])
        report['summary'] = {name: {'mean_video_auc': float(np.mean([report['seeds'][str(seed)][name]['video_auc'] for seed in SEEDS])),
                                   'std_video_auc': float(np.std([report['seeds'][str(seed)][name]['video_auc'] for seed in SEEDS], ddof=1))} for name in names}
        p.dump(report_path, report); all_metrics[dataset] = report
        p.log('V4_DATASET_COMPLETE', dataset=dataset, summary=report['summary'])
    if any(p.digest(path) != h for path, h in hashes.items()) or identity(config) != ident:
        raise ValueError('Immutable code/checkpoint identity changed')
    p.dump(output/'ALL_METRICS.json', all_metrics)
    p.dump(output/'COMPLETE.json', {'time': p.stamp(), 'datasets': datasets, 'identity': ident})
    return all_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    lease = p.acquire_gpu()
    root = Path(args.output).resolve()
    config = config_v4(); ident = identity(config)
    if root.exists():
        if not args.resume:
            raise ValueError('New output directory required')
        if json.loads((root/'IDENTITY.json').read_text()) != ident:
            raise ValueError('Code identity differs on resume')
    else:
        root.mkdir(parents=True)
        p.dump(root/'CONFIG.json', config); p.dump(root/'IDENTITY.json', ident)
    try:
        # All source partitions and hashes are verified before fitting.
        rows = p.manifests(config)
        data_selection = {name: {'frames': len(part), 'videos': len(p.groups(part)),
                          'families': len({r['family_id'] for r in part})} for name, part in rows.items()}
        if (root/'DATA_SELECTION.json').exists():
            if json.loads((root/'DATA_SELECTION.json').read_text()) != data_selection:
                raise ValueError('Data selection differs on resume')
        else:
            p.dump(root/'DATA_SELECTION.json', data_selection)
        all_training(root, config, rows, ident)
        reports = evaluate_all(root, config, ident)
        full = {name: report['summary']['Full'] for name, report in reports.items()}
        p.dump(root/'FINAL_REPORT.json', {'time': p.stamp(), 'selected': json.loads((root/'JOINT_SELECTION.json').read_text()),
               'full_auc': full, 'source_goal_met': full['source_val']['mean_video_auc'] >= .95,
               'each_target_goal_met': {name: full[name]['mean_video_auc'] >= .85 for name in config['targets']},
               'targets_exploratory': True, 'ablation_rankings_preserved': True})
        status(root, 'completed', status='completed')
    except Exception as error:
        status(root, 'failed', status='failed', error=repr(error)); raise


if __name__ == '__main__':
    main()
