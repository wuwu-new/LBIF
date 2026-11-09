"""V17 metadata/statistics only. Safe to import locally; no model libraries."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

RUN_NAME = 'v17_patch_forensics_20261101'
REMOTE_ROOT = '/root/autodl-tmp/drf_dual_transfer_20261026'
ARMS = ('UniformPatch', 'SoftForensics')
TARGETS = ('cdf1', 'cdf2', 'dfd', 'dfdc', 'dfdcp', 'wild', 'uadfv')
SEED = 2027
ANCHOR_SHA = 'aa7dc8cb2dc13edbc85abed47e815687ce14e9476286d02a21f5cc1b2d175865'
REFERENCE_SHA = 'b5c3ab1fb495f9362e4f0614826f2289f3bf7f6840351ccbeed2102ef5bc4282'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def configuration():
    return dict(seed=SEED, arms=list(ARMS), epochs=10, batch_size=16, workers=2,
                auxiliary_weight=.1, contrast_temperature=.1, likelihood_temperature=.1,
                patches_per_image=16, projection_dim=64, minimum_likelihood=.05,
                prototype='stop-gradient leave-video-out; equal available fake methods',
                weighting='fake patches only; mean-one per image; real patches uniform',
                patch_objective='symmetric cross-view class-positive log-ratio; no same-class negatives',
                spatial_correspondence='same normalized patch coordinates; no crop/flip in inherited views',
                auxiliary_video_deduplication='first sampled frame per video in each primary batch',
                primary_sampling='unchanged V5 WeightedRandomSampler replacement; real 1/2, four fake methods 1/8',
                model='V5 IR_LayerScale, semantic shortcut retained, g=1',
                semantic_sha256=ANCHOR_SHA, reference_sha256=REFERENCE_SHA,
                endpoint_rule='primary earliest best source VAL AUC among epochs 0..10; also report epoch10',
                original_training_repeated=False, source_test_access=False, targets_access=True,
                target_order=list(TARGETS), target_selection=False,
                automatic_expansion=False, promotion=False, memory_reserve_gib=6, disk_reserve_gib=1,
                degraded_source='fixed resize192-blur0.35-JPEG85, diagnostic only')


def unique_video_positions(rows):
    seen, positions = set(), []
    for i, row in enumerate(rows):
        if row['video_id'] not in seen:
            seen.add(row['video_id'])
            positions.append(i)
    return positions


def patch_positions(epoch, step):
    rng = random.Random(f'v17:{SEED}:{epoch}:{step}')
    return sorted(rng.sample(range(256), configuration()['patches_per_image']))


def verify_split(train, val):
    for rows, frames, videos, partition in ((train, 11492, 2873, 'fit'), (val, 22080, 690, 'source_val')):
        require(len(rows) == frames and len({r['video_id'] for r in rows}) == videos, 'Changed data denominator')
        require(all(r['partition'] == partition for r in rows), 'Wrong source partition')
        require(len({fingerprint([r['image_path'], r['sha256']]) for r in rows}) == frames, 'Repeated frame')
    for field in ('family_id', 'video_id', 'sha256'):
        require(not ({r[field] for r in train} & {r[field] for r in val}), 'TRAIN/VAL overlap: '+field)
    require({r['method'] for r in train if r['label']} == {'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'}, 'Changed methods')
    require(all(r['label'] == (0 if r['method'] == 'real' else 1) for r in train), 'Method/label mismatch')
    return {'train_frames': len(train), 'train_videos': 2873, 'train_families': len({r['family_id'] for r in train}),
            'val_frames': len(val), 'val_videos': 690, 'val_families': len({r['family_id'] for r in val}),
            'historical_semantic_saw_train': True, 'identity_disjointness_claimed': False}


def auc(labels, scores, weights=None):
    require(len(labels) == len(scores) and all(math.isfinite(x) for x in scores), 'Invalid scores')
    weights = [1]*len(labels) if weights is None else weights
    nr, nf = (sum(w for y, w in zip(labels, weights) if y == cls) for cls in (0, 1))
    if not nr or not nf:
        return None
    order = sorted(range(len(labels)), key=scores.__getitem__)
    preceding = credit = 0.
    start = 0
    while start < len(order):
        end = start+1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        real = sum(weights[i] for i in order[start:end] if not labels[i])
        fake = sum(weights[i] for i in order[start:end] if labels[i])
        credit += fake*(preceding+real/2)
        preceding += real
        start = end
    return credit/(nr*nf)


def paired(rows, base, candidate, replicates=1000, scope='reused source VAL, fixed single seed, no multiplicity/seed-population coverage'):
    labels = [r['label'] for r in rows]
    blocks = defaultdict(list)
    for i, row in enumerate(rows):
        blocks[row.get('family_id') or row['video_id']].append(i)
    blocks = list(blocks.values())
    rng, draws = random.Random(17012027), []
    for _ in range(replicates):
        weights = [0]*len(rows)
        for _ in blocks:
            for i in blocks[rng.randrange(len(blocks))]:
                weights[i] += 1
        a, b = auc(labels, base, weights), auc(labels, candidate, weights)
        if a is not None and b is not None:
            draws.append(100*(b-a))
    require(len(draws) >= .95*replicates, 'Too many invalid bootstrap replicates')
    draws.sort()
    def quantile(q):
        k = q*(len(draws)-1)
        i = int(k)
        return draws[i]+(draws[min(i+1, len(draws)-1)]-draws[i])*(k-i)
    repair = damage = 0.
    for f in (i for i, y in enumerate(labels) if y):
        for r in (i for i, y in enumerate(labels) if not y):
            old = (base[f] > base[r])-(base[f] < base[r])
            new = (candidate[f] > candidate[r])-(candidate[f] < candidate[r])
            change = (new-old)/2
            repair += max(0, change)
            damage += max(0, -change)
    denominator = labels.count(0)*labels.count(1)
    net = 100*(auc(labels, candidate)-auc(labels, base))
    require(abs(net-100*(repair-damage)/denominator) < 1e-10, 'Rank identity failed')
    return dict(net_pp=net, repair_pp=100*repair/denominator, damage_pp=100*damage/denominator,
                descriptive_95_ci_pp=[quantile(.025), quantile(.975)], replicates=len(draws), clusters=len(blocks),
                scope=scope)
