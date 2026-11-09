"""Stream the frozen GPU experts; no checkpoint/gate selection on targets."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import gpu_pipeline as pipeline
from model import GuardedDual, ReliabilityGate
from cache import Pixels
from prepare import digest, dump


def load_all(run, config, identity):
    anchor = pipeline.SemanticTokens(config['semantic_checkpoint']).cuda()
    models, gates = {}, {}
    for name in config['ablations']:
        folder = run / name
        state = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
        if state['identity'] != identity:
            raise ValueError('Checkpoint identity differs')
        model = GuardedDual(copy.deepcopy(anchor), state['spec']).cuda()
        model.load_state_dict(state['model'], strict=True)
        models[name] = model.eval()
        selected = {}
        center, scale = None, None
        for mode in ['sample', 'global']:
            gate_state = torch.load(folder / (mode+'_gate.pt'), map_location='cpu', weights_only=True)
            if gate_state['expert_sha256'] != digest(folder / 'best.pt'):
                raise ValueError('Gate/expert mismatch')
            gate = ReliabilityGate(mode).cuda()
            gate.load_state_dict(gate_state['state'], strict=True)
            selected[mode] = gate.eval()
            if center is not None:
                torch.testing.assert_close(center, gate_state['center'].cuda(), rtol=0, atol=0)
                torch.testing.assert_close(scale, gate_state['scale'].cuda(), rtol=0, atol=0)
            center, scale = gate_state['center'].cuda(), gate_state['scale'].cuda()
        gates[name] = selected, center, scale
    return models, gates


@torch.no_grad()
def stream_predictions(encoder, models, rows, config, dataset):
    result = {name: {k: [] for k in ['s', 'd', 'r']} for name in models}
    loader = DataLoader(Pixels(rows), batch_size=config['batch_size'], num_workers=config['workers'], pin_memory=True)
    done = 0
    started = time.monotonic()
    for rgb in loader:
        tokens = pipeline.encode(encoder, rgb)
        x = rgb.cuda(non_blocking=True)
        for name, model in models.items():
            out = model(x, tokens)
            for k, field in [('s', 'semantic_logit'), ('d', 'delta'), ('r', 'residual_logit')]:
                values = out.get(field, torch.zeros_like(out['logit'])).float()
                if not torch.isfinite(values).all():
                    raise ValueError('Nonfinite evaluation prediction')
                result[name][k].extend(values.cpu().tolist())
        done += len(rgb)
        if done % 2048 < len(rgb) or done == len(rows):
            pipeline.log('EVAL_PROGRESS', dataset=dataset, frames=done, total=len(rows), seconds=time.monotonic()-started)
    if done != len(rows):
        raise ValueError('Incomplete frame coverage')
    return {name: {k: np.asarray(v, np.float32) for k, v in pred.items()} for name, pred in result.items()}


def paired_source_intervals(rows, scores):
    ix = pipeline.groups(rows)
    labels = np.asarray([rows[x[0]]['label'] for x in ix])
    families = np.asarray([rows[x[0]]['family_id'] for x in ix])
    unique = sorted(set(families))
    grouped = [np.flatnonzero(families == family) for family in unique]
    video = {k: np.asarray([v[x].mean() for x in ix]) for k, v in scores.items()}
    generator = np.random.default_rng(2027)
    differences = {k: [] for k in ['NoResidual', 'NoGate', 'global']}
    for _ in range(1000):
        selection = np.concatenate([grouped[i] for i in generator.integers(0, len(grouped), len(grouped))])
        y = labels[selection]
        if len(set(y)) < 2:
            continue
        full = roc_auc_score(y, video['sample'][selection])
        for key in differences:
            differences[key].append(float(full - roc_auc_score(y, video[key][selection])))
    return {key: {'full_minus_control_auc': float(roc_auc_score(labels, video['sample'])-roc_auc_score(labels, video[key])),
                  'family_cluster_95_ci': np.quantile(values, [.025, .975]).tolist(),
                  'replicates': len(values), 'scope': 'one inherited seed; exploratory source-family uncertainty'}
            for key, values in differences.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['source', 'targets'])
    parser.add_argument('run')
    args = parser.parse_args()
    torch.set_num_threads(4)
    lease = pipeline.acquire_gpu()
    run = Path(args.run)
    if json.loads((run / 'STATE.json').read_text())['status'] != 'completed':
        raise ValueError('Source pilot incomplete')
    config = json.loads((run / 'CONFIG.json').read_text())
    identity = json.loads((run / 'IDENTITY.json').read_text())
    if pipeline.identities(config) != identity:
        raise ValueError('Frozen code/backbone differs')
    source = pipeline.ROOT / 'guarded_dual_20261025/run_v1/manifests'
    names = ['source_val'] if args.phase == 'source' else ['source_test', *config['targets']]
    lock = {'phase': args.phase, 'time': pipeline.stamp(), 'source_identity': identity,
            'evaluator_sha256': digest(__file__), 'config_sha256': digest(run/'CONFIG.json'),
            'checkpoints_and_gates': {str(p): digest(p) for p in sorted(run.glob('*/*.pt'))},
            'manifests': {name: digest(source / (name+'.json')) for name in names},
            'selection': 'all experts/gates fixed before evaluation; no per-target choices',
            'scope': 'conditional transfer pilot; not independent three-seed confirmation'}
    output = run / (args.phase+'_evaluation')
    output.mkdir(exist_ok=False)
    dump(output / 'EVAL_LOCK.json', lock)
    models, gates = load_all(run, config, identity)
    encoder = pipeline.load_encoder(config)
    all_metrics = {}
    for name in names:
        rows = json.loads((source / (name+'.json')).read_text())
        if any(len(ids) != 32 for ids in pipeline.groups(rows)):
            raise ValueError('Dataset is not complete exact32')
        missing = [r['image_path'] for r in rows if not Path(r['image_path']).is_file()]
        if missing:
            raise ValueError('Missing target/source frames: ' + str(missing[:3]))
        pred = stream_predictions(encoder, models, rows, config, name)
        report = {'frames': len(rows), 'videos': len(pipeline.groups(rows)), 'models': {}}
        for group, prediction in pred.items():
            report['models'][group], scores = pipeline.gated_scores(config, rows, prediction, *gates[group], diagnostic=args.phase=='source')
            pipeline.save_scores(output / (name+'_'+group+'.jsonl'), rows, prediction, scores)
            if args.phase == 'source' and group == 'Full':
                report['full_paired_intervals'] = paired_source_intervals(rows, scores)
        dump(output / (name+'_METRICS.json'), report)
        all_metrics[name] = report
        pipeline.log('DATASET_COMPLETE', dataset=name,
                     full_video_auc=report['models']['Full']['sample']['video_auc'],
                     semantic_video_auc=report['models']['NoResidual']['sample']['video_auc'],
                     nogate_video_auc=report['models']['Full']['NoGate']['video_auc'])
    if pipeline.identities(config) != identity or digest(__file__) != lock['evaluator_sha256']:
        raise ValueError('Evaluation identities changed')
    for p, h in lock['checkpoints_and_gates'].items():
        if digest(p) != h:
            raise ValueError('Checkpoint changed during evaluation')
    dump(output / 'ALL_METRICS.json', all_metrics)
    dump(output / 'STATE.json', {'status': 'completed', 'time': pipeline.stamp(),
                                'datasets': names, 'no_per_target_selection': True})


if __name__ == '__main__':
    main()
