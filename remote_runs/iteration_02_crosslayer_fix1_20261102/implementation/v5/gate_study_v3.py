"""CUDA-only gate study on immutable V2 expert predictions.

The two candidates are the semantic exit and the semantic-plus-correction exit.
This is dense post-expert routing, not sparse MoE dispatch or a speedup claim.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

import gpu_pipeline as pipeline


SPECS = {
    'BCE_SelectBCE': {'features': 'legacy', 'router': 'linear', 'objective': 'bce', 'select': 'bce'},
    'BCE_SelectAUC': {'features': 'legacy', 'router': 'linear', 'objective': 'bce', 'select': 'auc'},
    'Rank_SelectAUC': {'features': 'legacy', 'router': 'linear', 'objective': 'rank', 'select': 'auc'},
    'InvariantRank': {'features': 'invariant', 'router': 'linear', 'objective': 'rank', 'select': 'auc'},
    'SoftmaxTop2': {'features': 'invariant', 'router': 'softmax2', 'objective': 'rank', 'select': 'auc'},
    'SigmoidTop2': {'features': 'invariant', 'router': 'sigmoid2', 'objective': 'rank', 'select': 'auc'},
    'Top1': {'features': 'invariant', 'router': 'top1', 'objective': 'rank', 'select': 'auc'},
    'Top1LossFree': {'features': 'invariant', 'router': 'top1', 'objective': 'rank', 'select': 'auc', 'balance': True},
}
CONTROLS = ['Semantic', 'NoGate', 'OriginalSample', 'OriginalGlobal']
STEPS, RATE, DECAY = 600, .01, .01
RANK_TEMPERATURE, BCE_WEIGHT, BIAS_RATE = .1, .1, .005


def load_predictions(path):
    by_video = {}
    with Path(path).open() as source:
        for line in source:
            row = json.loads(line)
            by_video.setdefault(row['video_id'], []).append(row)
    ordered = sorted(by_video)
    rows = [by_video[key] for key in ordered]
    for video in rows:
        if len(video) != 32 or len({r['image_path'] for r in video}) != 32:
            raise ValueError('Expected 32 unique frames per video')
        if len({r['label'] for r in video}) != 1:
            raise ValueError('Inconsistent video labels')
    result = {key: torch.tensor([[r[key] for r in video] for video in rows],
                               dtype=torch.float32, device='cuda') for key in ['s', 'd', 'r']}
    result['y'] = torch.tensor([video[0]['label'] for video in rows], dtype=torch.float32, device='cuda')
    result['video_id'] = ordered
    result['family_id'] = [video[0].get('family_id', video[0]['video_id']) for video in rows]
    if any(not torch.isfinite(result[k]).all() for k in ['s', 'd', 'r', 'y']):
        raise ValueError('Nonfinite saved expert predictions')
    return result


def entropy(p):
    q = p.clamp(1e-6, 1 - 1e-6)
    return -(q * q.log() + (1 - q) * (1 - q).log())


def features(data, mode):
    s, d, r = (data[k] for k in ['s', 'd', 'r'])
    if mode == 'legacy':
        return pipeline.gate_features(s, d, r)
    p, q = s.sigmoid(), r.sigmoid()
    # Invariant to flipping BOTH class orientations: (s,d,r) -> (-s,-d,-r).
    # This removes signed class-score coordinates, not every possible class cue.
    return torch.stack([entropy(p).mean(-1), p.std(-1, unbiased=False),
                        entropy(q).mean(-1), q.std(-1, unbiased=False),
                        d.abs().mean(-1), d.std(-1, unbiased=False),
                        (p-q).abs().mean(-1), ((s+d).sigmoid()-p).abs().mean(-1)], -1).detach()


def video_probabilities(data, gate):
    return (data['s'] + gate[:, None] * data['d']).sigmoid().mean(-1)


def auc(y, scores):
    return float(roc_auc_score(y.detach().cpu().numpy(), scores.detach().cpu().numpy()))


def loss_value(data, gate, objective):
    p = video_probabilities(data, gate)
    bce = pipeline.balanced_video_bce(p, data['y'])
    if objective == 'bce':
        return bce
    positive, negative = p[data['y'] == 1], p[data['y'] == 0]
    rank = F.softplus(-(positive[:, None] - negative[None, :]) / RANK_TEMPERATURE).mean()
    return rank + BCE_WEIGHT * bce


class Router(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.layer = nn.Linear(8, 1 if kind == 'linear' else 2)
        if kind == 'linear':
            nn.init.zeros_(self.layer.weight)
            nn.init.zeros_(self.layer.bias)
        else:
            generator = torch.Generator().manual_seed(2027)
            with torch.no_grad():
                self.layer.weight.copy_(torch.randn(self.layer.weight.shape, generator=generator) * .01)
                self.layer.bias.zero_()
        self.register_buffer('expert_bias', torch.zeros(2))

    def forward(self, x, training_surrogate=False):
        logits = self.layer(x)
        if self.kind == 'linear':
            return logits[:, 0].sigmoid()
        if self.kind == 'sigmoid2':
            affinity = logits.sigmoid()
            # Two candidates and k=2: both are active; this is not sparse routing.
            weights = affinity / affinity.sum(-1, keepdim=True).clamp_min(1e-6)
            return weights[:, 1]
        soft = logits.softmax(-1)
        if self.kind == 'softmax2':
            return soft[:, 1]
        # The balancing bias changes selection only, never expert output weights.
        chosen = (soft + self.expert_bias).argmax(-1)
        hard = (chosen == 1).to(soft.dtype)
        if training_surrogate:
            return hard.detach() + soft[:, 1] - soft[:, 1].detach()
        return hard

    @torch.no_grad()
    def rebalance(self, x):
        selection = (self.layer(x).softmax(-1) + self.expert_bias).argmax(-1)
        load = torch.bincount(selection, minlength=2).float() / len(selection)
        self.expert_bias.add_(BIAS_RATE * torch.sign(.5 - load))
        self.expert_bias.sub_(self.expert_bias.mean())


def fit_one(name, spec, fit, holdout, output):
    raw = features(fit, spec['features'])
    center = raw.mean(0)
    scale = raw.std(0, unbiased=False).clamp_min(.05)
    fx = (raw - center) / scale
    hx = (features(holdout, spec['features']) - center) / scale
    model = Router(spec['router']).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=RATE, weight_decay=DECAY)
    best_key, best_state, best_step = None, None, None
    history = []
    started = time.monotonic()
    for step in range(STEPS + 1):
        if step:
            optimizer.zero_grad(set_to_none=True)
            gate = model(fx, training_surrogate=True)
            loss = loss_value(fit, gate, spec['objective'])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite gate objective')
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise ValueError('Nonfinite router gradient')
            optimizer.step()
            if spec.get('balance'):
                model.rebalance(fx)
        if step % 10 == 0:
            with torch.no_grad():
                gate = model(hx)
                p = video_probabilities(holdout, gate)
                a, b = auc(holdout['y'], p), float(pipeline.balanced_video_bce(p, holdout['y']))
            key = (a, -b) if spec['select'] == 'auc' else (-b, a)
            history.append({'step': step, 'holdout_auc': a, 'holdout_balanced_bce': b,
                            'holdout_gate_mean': float(gate.mean()),
                            'expert_bias': model.expert_bias.cpu().tolist()})
            if best_key is None or key > best_key:
                best_key, best_state, best_step = key, copy.deepcopy(model.state_dict()), step
    model.load_state_dict(best_state)
    model.eval()
    torch.save({'spec': spec, 'state': best_state, 'center': center.cpu(), 'scale': scale.cpu(),
                'selected_step': best_step, 'holdout_key': best_key}, output / (name + '.pt'))
    pipeline.dump(output / (name + '_HISTORY.json'), history)
    pipeline.log('GATE_STUDY_FIT', name=name, selected_step=best_step, seconds=time.monotonic()-started)
    return {'model': model, 'center': center, 'scale': scale, 'spec': spec}


@torch.no_grad()
def gate_for(name, data, fitted):
    if name in ['Semantic', 'NoGate']:
        return torch.full_like(data['y'], float(name == 'NoGate'))
    info = fitted[name]
    x = features(data, info['spec']['features'])
    return info['model']((x-info['center'])/info['scale'])


@torch.no_grad()
def measure(data, gate):
    p = video_probabilities(data, gate)
    result = {'video_auc': auc(data['y'], p),
              'balanced_video_bce': float(pipeline.balanced_video_bce(p, data['y'])),
              'video_log_loss': float(F.binary_cross_entropy(p.clamp(1e-6, 1-1e-6), data['y'])),
              'videos': len(p), 'frames': int(data['s'].numel()),
              'gate_mean': float(gate.mean()), 'gate_std': float(gate.std(unbiased=False)),
              'gate_quantiles': torch.quantile(gate, gate.new_tensor([0, .1, .5, .9, 1])).cpu().tolist(),
              'gate_mean_real': float(gate[data['y'] == 0].mean()),
              'gate_mean_fake': float(gate[data['y'] == 1].mean()),
              'fraction_gate_below_half': float((gate < .5).float().mean()),
              'mean_delta': float(data['d'].mean()), 'std_delta': float(data['d'].std(unbiased=False))}
    return result, p


@torch.no_grad()
def diagnostics(data, fitted, reference):
    ys = data['y']
    original = gate_for('OriginalSample', data, fitted)
    s = video_probabilities(data, torch.zeros_like(ys))
    full = video_probabilities(data, torch.ones_like(ys))
    loss_s = F.binary_cross_entropy(s.clamp(1e-6,1-1e-6), ys, reduction='none')
    loss_f = F.binary_cross_entropy(full.clamp(1e-6,1-1e-6), ys, reduction='none')
    beneficial = loss_f < loss_s
    gen = torch.Generator().manual_seed(2944)
    shuffled = original[torch.randperm(len(ys), generator=gen).cuda()]
    if reference is not None:
        for candidate, mode in [('OriginalSample', 'sample'), ('OriginalGlobal', 'global'), ('NoGate', 'NoGate')]:
            measured = measure(data, gate_for(candidate,data,fitted))[0]
            if abs(measured['video_auc'] - reference[mode]['video_auc']) > 1e-12:
                raise ValueError('Legacy prediction reproduction failed: ' + candidate)
    flip = {**data, **{k: -data[k] for k in ['s','d','r']}}
    torch.testing.assert_close(features(data,'invariant'),features(flip,'invariant'),rtol=1e-5,atol=1e-6)
    return {'legacy_reference_auc_exact': reference is not None,
            'legacy_gate_by_class': {'real': float(original[ys==0].mean()), 'fake': float(original[ys==1].mean())},
            'legacy_gate_if_dual_bce_better': float(original[beneficial].mean()) if beneficial.any() else None,
            'legacy_gate_if_dual_bce_worse': float(original[~beneficial].mean()) if (~beneficial).any() else None,
            'fraction_dual_bce_better': float(beneficial.float().mean()),
            'legacy_mean_gate_control': measure(data,original.mean().expand_as(original))[0],
            'legacy_shuffled_gate_control': measure(data,shuffled)[0],
            'orientation_invariant_features_gpu_check': True,
            'interpretation': 'source sensitivity and associations; not proof of causal failure or independent generalization'}


@torch.no_grad()
def evaluate(name, data, names, fitted, output):
    report, predictions, gates = {}, {}, {}
    for candidate in names:
        gate = gate_for(candidate, data, fitted)
        report[candidate], score = measure(data, gate)
        predictions[candidate], gates[candidate] = score.cpu().tolist(), gate.cpu().tolist()
    pipeline.dump(output / (name + '_METRICS.json'), report)
    with (output / (name + '_VIDEO_PREDICTIONS.jsonl')).open('x') as f:
        for i, video in enumerate(data['video_id']):
            f.write(json.dumps({'video_id': video, 'family_id': data['family_id'][i],
                                'label': int(data['y'][i]),
                                'probability': {k: v[i] for k,v in predictions.items()},
                                'gate': {k:v[i] for k,v in gates.items()}},allow_nan=False)+'\n')
    return report


def profile_routers(data, fitted):
    result = {}
    for name, item in fitted.items():
        x = (features(data,item['spec']['features'])-item['center'])/item['scale']
        model = item['model']
        with torch.no_grad():
            for _ in range(20): model(x)
            start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(200): model(x)
            end.record(); torch.cuda.synchronize()
        result[name]={'cuda_router_ms_per_batch':start.elapsed_time(end)/200,
                      'videos_in_batch':len(x),'parameters':sum(p.numel() for p in model.parameters()),
                      'scope':'router only, excludes feature construction and both image experts; no sparse execution implemented'}
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run')
    parser.add_argument('output')
    args=parser.parse_args()
    torch.set_num_threads(4)
    lease=pipeline.acquire_gpu()
    run,output=Path(args.run),Path(args.output)
    for p in [run/'STATE.json',run/'source_evaluation/STATE.json',run/'targets_evaluation/STATE.json']:
        if json.loads(p.read_text())['status']!='completed':
            raise ValueError('All preceding experiments must finish first')
    config=json.loads((run/'CONFIG.json').read_text())
    paths={'gate_fit':run/'Full/gate_fit_gate_predictions.jsonl',
           'gate_holdout':run/'Full/gate_holdout_gate_predictions.jsonl',
           'source_val':run/'source_evaluation/source_val_Full.jsonl'}
    targets=['source_test',*config['targets']]
    paths.update({n:run/'targets_evaluation'/(n+'_Full.jsonl') for n in targets})
    lock={'time':pipeline.stamp(),'script_sha256':pipeline.digest(__file__),
          'expert_sha256':pipeline.digest(run/'Full/best.pt'),
          'prediction_inputs':{k:{'path':str(p),'sha256':pipeline.digest(p)} for k,p in paths.items()},
          'original_gate_sha256':{mode:pipeline.digest(run/'Full'/(mode+'_gate.pt')) for mode in ['sample','global']},
          'specs':SPECS,'controls':CONTROLS,'steps':STEPS,'lr':RATE,'weight_decay':DECAY,
          'rank_temperature':RANK_TEMPERATURE,'rank_bce_weight':BCE_WEIGHT,'load_bias_rate':BIAS_RATE,
          'top1_bias_policy':'add bias to softmax affinities for selection only; update toward50/50 training loads; freeze at evaluation',
          'source_selection':'per-candidate checkpoint on gate_holdout; final candidate by source_val video AUC; ties prefer NoGate',
          'target_policy':'all candidates frozen together; report all; no per-target selection or refitting',
          'scope':'exploratory gate-only followup AFTER V2 target observation; inherited expert, no fresh-seed claim',
          'compute':'all train/inference tensors on CUDA; saved frozen expert outputs reused; both branches already computed, no dispatch or communication savings'}
    output.mkdir(exist_ok=False)
    pipeline.dump(output/'PREREG.json',lock)
    try:
        data={k:load_predictions(paths[k]) for k in ['gate_fit','gate_holdout','source_val']}
        for a,b in [('gate_fit','gate_holdout'),('gate_fit','source_val'),('gate_holdout','source_val')]:
            if set(data[a]['family_id']) & set(data[b]['family_id']):
                raise ValueError('Source families overlap')
        fitted={}
        for candidate,mode in [('OriginalSample','sample'),('OriginalGlobal','global')]:
            state=torch.load(run/'Full'/(mode+'_gate.pt'),map_location='cpu',weights_only=True)
            if state['expert_sha256']!=lock['expert_sha256']: raise ValueError('Legacy gate/expert mismatch')
            model=pipeline.ReliabilityGate(mode).cuda(); model.load_state_dict(state['state'],strict=True); model.eval()
            fitted[candidate]={'model':model,'center':state['center'].cuda(),'scale':state['scale'].cuda(),'spec':{'features':'legacy'}}
        for name,spec in SPECS.items():
            fitted[name]=fit_one(name,spec,data['gate_fit'],data['gate_holdout'],output)
        names=CONTROLS+list(SPECS)
        source={k:evaluate(k,v,names,fitted,output) for k,v in data.items()}
        ref=json.loads((run/'Full/SOURCE_GATE_REPORT.json').read_text())['source_val']
        diag=diagnostics(data['source_val'],fitted,ref)
        pipeline.dump(output/'SOURCE_DIAGNOSTICS.json',diag)
        pipeline.dump(output/'ROUTER_TIMINGS.json',profile_routers(data['source_val'],fitted))
        if abs(source['source_val']['BCE_SelectBCE']['video_auc']-source['source_val']['OriginalSample']['video_auc'])>1e-12:
            raise ValueError('BCE reproduction AUC differs; preserve study for diagnosis')
        winner=max(names,key=lambda n:(source['source_val'][n]['video_auc'],n=='NoGate'))
        frozen={str(p):pipeline.digest(p) for p in sorted(output.glob('*.pt'))}
        selection={'winner':winner,'rule':lock['source_selection'],'source_val_auc':source['source_val'][winner]['video_auc'],
                   'source_only':True,'no_target_selection':True,'frozen_router_sha256':frozen,
                   'warning':'source VAL also selected the inherited V2 expert; this is not independent confirmation'}
        pipeline.dump(output/'SELECTION.json',selection)
        pipeline.log('GATE_STUDY_SOURCE_COMPLETE',**selection)
        reports={}
        for name in targets:
            value=load_predictions(paths[name])
            reports[name]=evaluate(name,value,names,fitted,output)
            pipeline.log('GATE_STUDY_TARGET',dataset=name,winner=winner,
                         selected_auc=reports[name][winner]['video_auc'],
                         legacy_auc=reports[name]['OriginalSample']['video_auc'],
                         nogate_auc=reports[name]['NoGate']['video_auc'])
        for p,h in frozen.items():
            if pipeline.digest(p)!=h: raise ValueError('Router changed during evaluation')
        for item in lock['prediction_inputs'].values():
            if pipeline.digest(item['path'])!=item['sha256']: raise ValueError('Frozen expert predictions changed')
        if pipeline.digest(run/'Full/best.pt')!=lock['expert_sha256'] or pipeline.digest(__file__)!=lock['script_sha256']:
            raise ValueError('Expert or script changed')
        pipeline.dump(output/'ALL_METRICS.json',{'source':source,'targets':reports,'selection':selection})
        pipeline.dump(output/'STATE.json',{'status':'completed','time':pipeline.stamp(),'device':torch.cuda.get_device_name()})
    except Exception as error:
        pipeline.dump(output/'STATE.json',{'status':'failed','time':pipeline.stamp(),'error':repr(error)})
        raise


if __name__=='__main__':
    main()
