"""Remote-only GPU pilot; source-only selection and immutable target reporting."""
import argparse
import collections
import copy
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import CLIPVisionModel


ROOT = Path('/root/autodl-tmp/drf_dual_transfer_20261026')
PRIOR = ROOT / 'guarded_dual_20261025/code'
LEGACY = ROOT / 'lijiale_correction_blend_20261026/code'
sys.path.insert(0, str(PRIOR))
sys.path.insert(0, str(LEGACY))
from model import GuardedDual, ReliabilityGate, gate_features, balanced_video_bce
from cache import Pixels, MEAN, STD
from prepare import digest, dump
from run import metrics, groups, check_ablation_plan
from drf.core.fusion import GatedResidualCrossAttention
from drf.core.heads import BinaryClassifierHead


def stamp():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def log(kind, **values):
    print(kind, json.dumps({'time': stamp(), **values}, allow_nan=False), flush=True)


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def acquire_gpu():
    lease = (ROOT / '.gpu_exclusive.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    probe = subprocess.run(['nvidia-smi', '--query-compute-apps=pid',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    others = [int(x.strip()) for x in probe.stdout.splitlines()
              if x.strip().isdigit() and int(x.strip()) != os.getpid()]
    if others:
        raise RuntimeError('Existing GPU users preserved: ' + str(others))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU fallback forbidden')
    return lease


def state_of(path):
    # Trusted, byte-verified project checkpoint, never an untrusted download.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    return checkpoint, checkpoint.get('ema_state') or checkpoint['model_state']


class SemanticTokens(nn.Module):
    """The complete historical NoResidual token classifier, not a weaker CLS head."""
    def __init__(self, checkpoint):
        super().__init__()
        self.semantic_query = nn.Linear(1024, 192)
        self.fusion = GatedResidualCrossAttention(192, 1024, 256, 8, 128, .1,
                                                 use_gate=True, use_residual_pool=True)
        self.classifier = BinaryClassifierHead(128, 64, .1)
        _, source = state_of(checkpoint)
        picked = {k: v for k, v in source.items()
                  if k.startswith(('semantic_query.', 'fusion.', 'classifier.'))}
        self.load_state_dict(picked, strict=True)
        self.requires_grad_(False).eval()

    def train(self, mode=True):
        return super().train(False)

    def forward(self, tokens):
        return self.classifier(self.fusion(self.semantic_query(tokens.float()), tokens.float()))


def configuration():
    config = json.loads((PRIOR / 'config.remote.json').read_text())
    config.update(name='guarded_transfer_pilot_20261026', seed=2027,
                  epochs=3, batch_size=16, workers=2, fit_families=32,
                  pilot_val_families=8, gate_families_per_fold=6,
                  gate_steps=300,
                  clip_path='/root/drf_selective_pilot_clip_20261026_01',
                  semantic_checkpoint=str(ROOT / 'checkpoints/no_residual/ckpt_best.pth'))
    return config


def identities(config):
    paths = [Path(__file__), *sorted(PRIOR.glob('*.py')),
             PRIOR / 'assets/teacher_SRM_Kernels.npy',
             LEGACY / 'drf/core/fusion.py', LEGACY / 'drf/core/heads.py',
             Path(config['semantic_checkpoint']), Path(config['clip_path']) / 'config.json',
             Path(config['clip_path']) / 'model.safetensors']
    return {str(p): digest(p) for p in paths}


def verify_source(config):
    reference = {
        'train': 'f57f61505262950a623f22f144a9f3c80d04f368c613b66cf7e187f1f5d558c1',
        'val': '6641a3f7e06ab85088803d900dc7bc05278855eeac5fbadbdf94e0c0a95cf23f',
        'test': 'ada1c6a6e30a90dfa8814225e56041b924402e5272fc012209f11d45cf1f0bc3'}
    for name, path in config['source_indexes'].items():
        if digest(path) != reference[name]:
            raise ValueError('Transferred source index differs: ' + name)
    result = {}
    manifests = ROOT / 'guarded_dual_20261025/run_v1/manifests'
    for name in ['fit', 'source_val', 'gate_fit', 'gate_holdout']:
        rows = json.loads((manifests / (name + '.json')).read_text())
        for row in rows:
            if not Path(row['image_path']).is_file():
                raise ValueError('Missing data: ' + row['image_path'])
        result[name] = {'rows': len(rows), 'videos': len(groups(rows)),
                        'families': len({r['family_id'] for r in rows})}
    return result


def subset(rows, count, namespace):
    families = sorted({r['family_id'] for r in rows},
                      key=lambda x: hashlib.sha256((namespace + x).encode()).hexdigest())[:count]
    keep = set(families)
    return [r for r in rows if r['family_id'] in keep]


def manifests(config):
    folder = ROOT / 'guarded_dual_20261025/run_v1/manifests'
    result = {name: json.loads((folder / (name + '.json')).read_text())
              for name in ['fit', 'source_val', 'gate_fit', 'gate_holdout']}
    for name, count in [('fit', config['fit_families']),
                        ('source_val', config['pilot_val_families']),
                        ('gate_fit', config['gate_families_per_fold']),
                        ('gate_holdout', config['gate_families_per_fold'])]:
        result[name] = subset(result[name], count, 'guarded-pilot-v1:' + name + ':')
    seen_family, seen_hash = {}, {}
    for name, rows in result.items():
        for row in rows:
            if digest(row['image_path']) != row['sha256']:
                raise ValueError('Selected source bytes changed')
            for value, lookup in [(row['family_id'], seen_family), (row['sha256'], seen_hash)]:
                if lookup.setdefault(value, name) != name:
                    raise ValueError('Source split overlap')
    return result


@torch.no_grad()
def encode(encoder, rgb):
    x = rgb.cuda(non_blocking=True)
    mean = x.new_tensor(MEAN)[None, :, None, None]
    std = x.new_tensor(STD)[None, :, None, None]
    with torch.autocast('cuda', dtype=torch.float16):
        tokens = encoder(pixel_values=(x - mean) / std).last_hidden_state
    if not torch.isfinite(tokens).all():
        raise ValueError('Nonfinite CLIP tokens')
    return tokens.to(torch.float16).float()


def load_encoder(config):
    return CLIPVisionModel.from_pretrained(config['clip_path'], local_files_only=True).cuda().eval().requires_grad_(False)


def gpu_audit(config, out):
    verification = verify_source(config)
    _, historical = state_of(config['semantic_checkpoint'])
    correction_path = ROOT / ('lijiale_correction_blend_20261026/runs/'
                             'drf_correction_blend_from_noresidual_seed2027/ckpt/ckpt_best.pth')
    checkpoint, corrected = state_of(correction_path)
    raw = float(corrected['blend_raw'])
    semantic_changed = [k for k, v in corrected.items()
                        if k.startswith(('semantic_query.', 'fusion.', 'classifier.'))
                        and k in historical and not torch.equal(v, historical[k])]
    semantic_max_drift = {k: float((corrected[k].float() - historical[k].float()).abs().max())
                          for k in semantic_changed}
    row = json.loads(Path(config['source_indexes']['val']).read_text())[0]
    encoder = load_encoder(config)
    pixels = Pixels([row, row])[0].unsqueeze(0).repeat(2, 1, 1, 1)
    tokens = encode(encoder, pixels)
    anchor = SemanticTokens(config['semantic_checkpoint']).cuda()
    corrected_semantic = SemanticTokens(correction_path).cuda()
    with torch.no_grad():
        semantic_logit_drift = float((anchor(tokens)-corrected_semantic(tokens)).abs().max())
    del corrected_semantic
    seed_all(config['seed'])
    model = GuardedDual(anchor, config['model']).cuda().train()
    with torch.no_grad():
        base = anchor(tokens)
    response = model(pixels.cuda(), tokens)
    torch.testing.assert_close(response['logit'], base, rtol=0, atol=0)
    loss = F.binary_cross_entropy_with_logits(response['logit'], torch.tensor([0., 1.], device='cuda'))
    loss = loss + .1 * F.binary_cross_entropy_with_logits(response['residual_logit'], torch.tensor([0., 1.], device='cuda'))
    loss.backward()
    no_residual = GuardedDual(copy.deepcopy(anchor), {**config['model'], 'use_residual': False}).cuda().eval()
    with torch.no_grad():
        torch.testing.assert_close(no_residual(pixels.cuda(), tokens)['logit'], base, rtol=0, atol=0)
    no_semantic = GuardedDual(copy.deepcopy(anchor), {**config['model'], 'use_semantic': False}).cuda().eval()
    with torch.no_grad():
        no_semantic.correction[-1].weight.normal_()
        torch.testing.assert_close(no_semantic(pixels.cuda(), tokens)['logit'],
                                   no_semantic(pixels.cuda(), tokens.flip(0)*3)['logit'], rtol=0, atol=0)
    checks = {
        'initial_full_exactly_semantic': True,
        'semantic_eval_locked': not anchor.training and not any(p.grad is not None for p in anchor.parameters()),
        'residual_aux_finite_nonzero_gradient': any(p.grad is not None and torch.isfinite(p.grad).all() and bool(p.grad.abs().sum())
                                                 for p in model.encoder.parameters()),
        'zero_init_correction_last_layer_gets_gradient': bool(model.correction[-1].weight.grad.abs().sum()),
        'source_index_and_frame_coverage': verification,
        'legacy_correction_gate_raw': raw,
        'legacy_correction_gate_clamped': min(1., max(0., raw)),
        'legacy_semantic_changed_tensor_names': semantic_changed,
        'legacy_semantic_max_parameter_drift': max(semantic_max_drift.values(), default=0.),
        'legacy_semantic_logit_drift_two_source_frames': semantic_logit_drift,
        'no_residual_exact_exit_gpu': True,
        'no_semantic_cannot_read_token_values_gpu': True,
        'legacy_selected_epoch': checkpoint.get('epoch'),
        'device': torch.cuda.get_device_name(),
        'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
        'torch_version': torch.__version__, 'scope': 'GPU contract check, NOT AUC evidence'}
    if not all(checks[k] for k in ['semantic_eval_locked', 'residual_aux_finite_nonzero_gradient',
                                 'zero_init_correction_last_layer_gets_gradient']):
        raise RuntimeError('GPU contract failed')
    dump(out / 'GPU_AUDIT.json', checks)
    log('GPU_AUDIT', **checks)
    del model, no_residual, no_semantic, anchor, encoder
    torch.cuda.empty_cache()


@torch.no_grad()
def memory_cache(encoder, rows, config, augmented=False):
    # Pilot-only RAM caches: no 10+ GB feature files on the nearly full data disk.
    tokens = torch.empty((len(rows), 257, 1024), dtype=torch.float16)
    pixels = torch.empty((len(rows), 3, 224, 224), dtype=torch.uint8)
    offset = 0
    loader = DataLoader(Pixels(rows, augmented), batch_size=config['batch_size'],
                        num_workers=config['workers'], pin_memory=True)
    for rgb in loader:
        n = len(rgb)
        value = encode(encoder, rgb).cpu().half()
        tokens[offset:offset+n] = value
        pixels[offset:offset+n] = (rgb * 255).round().to(torch.uint8)
        offset += n
        if offset % 512 < n or offset == len(rows):
            log('RAM_CACHE', augmented=augmented, done=offset, total=len(rows))
    return {'tokens': tokens, 'rgb': pixels}


@torch.no_grad()
def predict(model, cache, batch_size):
    result = {k: [] for k in ['s', 'd', 'r']}
    model.eval()
    for start in range(0, len(cache['tokens']), batch_size):
        t = cache['tokens'][start:start+batch_size].cuda().float()
        rgb = cache['rgb'][start:start+batch_size].cuda().float() / 255
        out = model(rgb, t)
        for key, field in [('s', 'semantic_logit'), ('d', 'delta'), ('r', 'residual_logit')]:
            result[key].extend(out.get(field, torch.zeros_like(out['logit'])).cpu().tolist())
    return {k: np.asarray(v, np.float32) for k, v in result.items()}


def probability(pred, gate=1.):
    return 1 / (1 + np.exp(-np.clip(pred['s'] + gate * pred['d'], -50, 50)))


def save_scores(out, rows, pred, variants=None):
    with out.open('x') as f:
        for i, row in enumerate(rows):
            value = {**row, **{k: float(v[i]) for k, v in pred.items()}}
            if variants is not None:
                value['probabilities'] = {k: float(v[i]) for k, v in variants.items()}
            f.write(json.dumps(value, allow_nan=False) + '\n')


def train_group(config, name, anchor, caches, rows, folder, identity):
    folder.mkdir(exist_ok=False)
    seed_all(config['seed'])
    spec = {**config['model'], **config['ablations'][name]}
    model = GuardedDual(copy.deepcopy(anchor), spec).cuda()
    parameters = [p for p in model.parameters() if p.requires_grad]
    filt = [model.front.delta] if model.front.delta.requires_grad else []
    ids = {id(p) for p in filt}
    opt = torch.optim.AdamW([
        {'params': [p for p in parameters if id(p) not in ids], 'lr': config['dual_lr']},
        {'params': filt, 'lr': config['dual_lr'] * config['filter_lr_multiplier']}],
        weight_decay=config['weight_decay'])
    fit = rows['fit']
    method_counts = collections.Counter(r['method'] for r in fit)
    fake_methods = {r['method'] for r in fit if r['label']}
    weights = [(.5 if not r['label'] else .5 / len(fake_methods)) / method_counts[r['method']] for r in fit]
    labels = torch.tensor([r['label'] for r in fit], dtype=torch.float32, device='cuda')
    best = -math.inf
    best_pred = None
    history = []
    for epoch in range(config['epochs'] + 1):
        started = time.monotonic()
        loss_sum = 0.
        updates = 0
        if epoch and spec['use_residual']:
            model.train()
            generator = torch.Generator().manual_seed(config['seed'] + 10000 * epoch)
            sampler = list(WeightedRandomSampler(weights, len(fit), True, generator=generator))
            for start in range(0, len(sampler), config['batch_size']):
                indexes = sampler[start:start+config['batch_size']]
                y = labels[indexes]
                opt.zero_grad(set_to_none=True)
                outs, losses = [], []
                for view in ['fit', 'fit_aug']:
                    t = caches[view]['tokens'][indexes].cuda().float()
                    rgb = caches[view]['rgb'][indexes].cuda().float() / 255
                    result = model(rgb, t)
                    loss = F.binary_cross_entropy_with_logits(result['logit'], y)
                    loss = loss + config['residual_aux_weight'] * F.binary_cross_entropy_with_logits(result['residual_logit'], y)
                    losses.append(loss)
                    outs.append(result)
                loss = torch.stack(losses).mean() + config['consistency_weight'] * F.mse_loss(outs[0]['delta'], outs[1]['delta'])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite loss')
                loss.backward()
                for p in parameters:
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        raise ValueError('Nonfinite gradient')
                nn.utils.clip_grad_norm_(parameters, 1.)
                opt.step()
                updates += 1
                loss_sum += float(loss.detach())
        pred = predict(model, caches['source_val'], config['batch_size'])
        score = metrics(rows['source_val'], probability(pred))
        entry = {'group': name, 'epoch': epoch, **score, 'updates': updates,
                 'loss': loss_sum / max(1, updates), 'seconds': time.monotonic() - started}
        history.append(entry)
        with (folder / 'history.jsonl').open('a') as f:
            f.write(json.dumps(entry) + '\n')
        if score['video_auc'] > best:
            best = score['video_auc']
            best_pred = pred
            torch.save({'model': model.state_dict(), 'spec': spec, 'epoch': epoch,
                        'score': score, 'identity': identity}, folder / 'best.pt')
        log('EPOCH', **entry)
        if not spec['use_residual']:
            break
    save_scores(folder / 'selected_pilot_val.jsonl', rows['source_val'], best_pred)
    state = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    dump(folder / 'SELECTION.json', {'epoch': state['epoch'], **state['score'],
        'rule': 'pilot source validation VIDEO AUC only; epoch0 eligible',
        'spec': spec, 'scope': 'conditional transfer pilot, not full-budget retraining'})
    return model


def video_tensors(rows, pred):
    ix = groups(rows)
    if any(len(x) != 32 for x in ix):
        raise ValueError('Exactly32 frames/video required')
    arrays = [torch.tensor(np.stack([pred[k][i] for i in ix]), device='cuda') for k in ['s', 'd', 'r']]
    y = torch.tensor([rows[i[0]]['label'] for i in ix], dtype=torch.float32, device='cuda')
    return (*arrays, y, ix)


def fit_gate(config, name, model, caches, rows, folder):
    predictions = {part: predict(model, caches[part], config['batch_size']) for part in ['gate_fit', 'gate_holdout', 'source_val']}
    fs, fd, fr, fy, _ = video_tensors(rows['gate_fit'], predictions['gate_fit'])
    hs, hd, hr, hy, _ = video_tensors(rows['gate_holdout'], predictions['gate_holdout'])
    fx, hx = gate_features(fs, fd, fr), gate_features(hs, hd, hr)
    center, scale = fx.mean(0), fx.std(0, unbiased=False).clamp_min(.05)
    selected = {}
    for mode in ['sample', 'global']:
        gate = ReliabilityGate(mode).cuda()
        opt = torch.optim.AdamW(gate.parameters(), lr=config['gate_lr'], weight_decay=.01)
        best_loss, best_state, best_step = math.inf, None, None
        for step in range(config['gate_steps'] + 1):
            if step:
                opt.zero_grad(set_to_none=True)
                p = (fs + gate((fx-center)/scale)[:, None] * fd).sigmoid().mean(-1)
                loss = balanced_video_bce(p, fy)
                loss.backward()
                opt.step()
            if step % 10 == 0:
                with torch.no_grad():
                    hp = (hs + gate((hx-center)/scale)[:, None] * hd).sigmoid().mean(-1)
                    loss = float(balanced_video_bce(hp, hy))
                if loss < best_loss:
                    best_loss, best_state, best_step = loss, copy.deepcopy(gate.state_dict()), step
        gate.load_state_dict(best_state)
        gate.eval()
        selected[mode] = gate
        torch.save({'state': best_state, 'center': center.cpu(), 'scale': scale.cpu(),
                    'step': best_step, 'holdout_balanced_video_bce': best_loss,
                    'expert_sha256': digest(folder / 'best.pt')}, folder / (mode+'_gate.pt'))
    report = {}
    for part, pred in predictions.items():
        report[part], probs = gated_scores(config, rows[part], pred, selected, center, scale, True)
        save_scores(folder / (part+'_gate_predictions.jsonl'), rows[part], pred, probs)
    dump(folder / 'SOURCE_GATE_REPORT.json', report)
    log('GATE', group=name, source_val=report['source_val'])
    return selected, center, scale


@torch.no_grad()
def gated_scores(config, rows, pred, selected, center, scale, diagnostic=False):
    s, d, r, y, ix = video_tensors(rows, pred)
    x = gate_features(s, d, r)
    gates = {'NoResidual': torch.zeros_like(y), 'NoGate': torch.ones_like(y)}
    gates.update({mode: gate((x-center)/scale) for mode, gate in selected.items()})
    if diagnostic:
        gates['SampleMean'] = gates['sample'].mean().expand_as(y)
        generator = torch.Generator().manual_seed(config['seed'] + 917)
        gates['SampleShuffled'] = gates['sample'][torch.randperm(len(y), generator=generator).cuda()]
    report, probabilities = {}, {}
    for key, g in gates.items():
        frame = (s + g[:, None]*d).sigmoid().cpu().numpy()
        scores = np.empty(len(rows), np.float32)
        for i, ids in enumerate(ix):
            scores[ids] = frame[i]
        probabilities[key] = scores
        report[key] = {**metrics(rows, scores), 'gate_mean': float(g.mean()),
                       'gate_std': float(g.std(unbiased=False)),
                       'balanced_video_bce': float(balanced_video_bce(torch.tensor(frame.mean(-1), device='cuda'), y))}
    return report, probabilities


def pilot(config, out, identity):
    rows = manifests(config)
    for part, selected in rows.items():
        dump(out / (part+'.json'), selected)
    dump(out / 'DATA_SELECTION.json', {k: {'frames': len(v), 'videos': len(groups(v)),
        'families': len({r['family_id'] for r in v})} for k, v in rows.items()})
    encoder = load_encoder(config)
    caches = {part: memory_cache(encoder, selected, config) for part, selected in rows.items()}
    caches['fit_aug'] = memory_cache(encoder, rows['fit'], config, augmented=True)
    del encoder
    torch.cuda.empty_cache()
    anchor = SemanticTokens(config['semantic_checkpoint']).cuda()
    group_names = list(config['ablations'])
    models, gates = {}, {}
    for name in group_names:
        folder = out / name
        model = train_group(config, name, anchor, caches, rows, folder, identity)
        common = {k: v for k, v in model.semantic.state_dict().items()}
        if any(not torch.equal(v, anchor.state_dict()[k]) for k, v in common.items()):
            raise ValueError('Semantic anchor changed during correction training')
        models[name] = model
        gates[name] = fit_gate(config, name, model, caches, rows, folder)
    dump(out / 'PILOT_COMPLETE.json', {'time': stamp(), 'groups': group_names,
        'targets_evaluated': False, 'source_selection_only': True,
        'scope': 'small family subset, inherited seed2027 semantic expert; not final95/85 evidence'})
    log('PILOT_COMPLETE', groups=group_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['audit', 'pilot'])
    parser.add_argument('output')
    args = parser.parse_args()
    torch.set_num_threads(4)
    lease = acquire_gpu()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    config = configuration()
    check_ablation_plan(config)
    free = shutil.disk_usage(ROOT).free
    if free < 2 * 1024**3:
        raise RuntimeError('Insufficient output space; existing files preserved')
    identity = identities(config)
    dump(out / 'IDENTITY.json', identity)
    dump(out / 'CONFIG.json', config)
    log('START', command=args.command, output=str(out), disk_free_bytes=free)
    try:
        gpu_audit(config, out)
        if args.command == 'pilot':
            pilot(config, out, identity)
        if identity != identities(config):
            raise ValueError('Source code/weights changed during execution')
        dump(out / 'STATE.json', {'status': 'completed', 'time': stamp()})
    except Exception as error:
        dump(out / 'STATE.json', {'status': 'failed', 'time': stamp(), 'error': repr(error)})
        raise


if __name__ == '__main__':
    main()
