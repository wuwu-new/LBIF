"""Create immutable source partitions and data audit, without model inference."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import struct


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def load_index(path):
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected nonempty frame list: ' + str(path))
    groups, identities = collections.defaultdict(list), set()
    for row in rows:
        if type(row.get('label')) is not int or row['label'] not in (0, 1):
            raise ValueError('Invalid binary label')
        p = Path(row['image_path'])
        if not p.is_absolute() or not p.is_file():
            raise ValueError('Missing frame: ' + str(p))
        key = (row['video_id'], str(p.resolve()))
        if key in identities:
            raise ValueError('Repeated frame identity')
        identities.add(key)
        groups[row['video_id']].append(dict(row))
    for vid, items in groups.items():
        if len({x['label'] for x in items}) != 1:
            raise ValueError('Video label collision: ' + vid)
        def order(x):
            numbers=re.findall(r'\d+',Path(x['image_path']).stem)
            return (int(x['source_frame_number']) if 'source_frame_number' in x else int(numbers[-1]) if numbers else -1,x['image_path'])
        items.sort(key=order)
    return groups


def take(items, frames):
    if len(items) < frames:
        raise ValueError('Not enough frames')
    return [dict(items[round(i * (len(items)-1) / (frames-1))]) for i in range(frames)]


def prepare(config, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    source = {split: load_index(path) for split, path in config['source_indexes'].items()}
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    for groups in source.values():
        for vid in groups:
            ids = vid.rsplit(':', 1)[-1].split('_')
            if any(not x.isdigit() for x in ids):
                raise ValueError('Cannot infer FF++ identity family: ' + vid)
            for x in ids[1:]:
                a, b = find(ids[0]), find(x)
                parent[max(a, b)] = min(a, b)
            find(ids[0])
    ownership = {}
    family_members = collections.defaultdict(set)
    for x in parent:
        family_members[find(x)].add(x)
    partitions = collections.defaultdict(list)
    source_stats = {}
    # Split membership is based only on original identities, never images/scores.
    for split, groups in source.items():
        for vid, items in groups.items():
            if len(items) != 32:
                raise ValueError('Fixed32 source index is incomplete')
            family = '_'.join(sorted(family_members[find(vid.rsplit(':', 1)[-1].split('_')[0])]))
            if family in ownership and ownership[family] != split:
                raise ValueError('Source/donor family crosses source splits')
            ownership[family] = split
            if split == 'train':
                h = int(hashlib.sha256(('drf-family-partition-v1:' + family).encode()).hexdigest()[:8], 16) % 100
                part = 'gate_holdout' if h < 10 else 'gate_fit' if h < 20 else 'fit'
                n = config['train_frames'] if part == 'fit' else 32
            else:
                part, n = 'source_' + split, 32
            for i, row in enumerate(take(items, n)):
                row.update(dataset_id='ffpp_c23', family_id=family, partition=part,
                           frame_index=i, method=vid.split(':')[1], sha256=digest(row['image_path']))
                partitions[part].append(row)
        source_stats[split] = {'videos': len(groups), 'frames': sum(map(len, groups.values())),
            'labels_by_video': dict(collections.Counter(v[0]['label'] for v in groups.values()))}
    # Exact selected-image hashes must not cross partitions either.
    seen_hashes = {}
    for part, rows in partitions.items():
        for row in rows:
            old = seen_hashes.setdefault(row['sha256'], part)
            if old != part:
                raise ValueError('Identical selected image crosses partitions: ' + old + '/' + part)
        if {row['label'] for row in rows} != {0, 1}:
            raise ValueError('Partition needs both classes: ' + part)
    output.mkdir(parents=True, exist_ok=False)
    for part, rows in partitions.items():
        dump(output / (part + '.json'), rows)
    targets = {}
    for name, item in config['targets'].items():
        groups = load_index(item['index'])
        selected, excluded = [], []
        for vid, rows in sorted(groups.items()):
            if len(rows) < 32:
                excluded.append({'video_id': vid, 'label': rows[0]['label'], 'available': len(rows)})
                continue
            for i, row in enumerate(take(rows, 32)):
                row.update(dataset_id=name, family_id=vid, frame_index=i, partition='target_test')
                selected.append(row)
        if {x['label'] for x in selected} != {0, 1}:
            raise ValueError('Target lacks both retained classes: ' + name)
        dump(output / (name + '.json'), selected)
        dump(output / (name + '_excluded.json'), excluded)
        samples = []
        for label in [0, 1]:
            candidates = sorted((x for x in selected if x['label'] == label),
                                key=lambda x: hashlib.sha256(x['image_path'].encode()).hexdigest())[:16]
            for row in candidates:
                with Path(row['image_path']).open('rb') as f:
                    h = f.read(24)
                samples.append({'path': row['image_path'], 'label': label,
                    'png_wh': struct.unpack('>II', h[16:24]) if h[:8] == b'\x89PNG\r\n\x1a\n' else None})
        targets[name] = {'input_index': item['index'], 'index_sha256': digest(item['index']),
            'protocol': item['protocol'], 'input_videos': len(groups), 'retained_videos': len(selected)//32,
            'retained_labels': dict(collections.Counter(x['label'] for x in selected[::32])),
            'excluded_videos': len(excluded), 'header_samples': samples,
            'scope': 'Retained subset: >=32 available frames, deterministic uniform selection; no relabelling'}
    audit = {'source': source_stats, 'partitions': {k: {'frames': len(v),
        'videos': len({x['video_id'] for x in v}), 'families': len({x['family_id'] for x in v})}
        for k,v in partitions.items()}, 'targets': targets,
        'split_policy': 'Original source/donor connected components, fixed hash, fit80/gate_fit10/gate_holdout10 nominal',
        'source_selected_image_hashes_disjoint': True,
        'target_policy': 'Evaluation only; mixed preprocessing protocols reported separately',
        'config_sha256': digest(config['_config_path'])}
    dump(output / 'DATA_AUDIT.json', audit)
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('config');p.add_argument('output');a=p.parse_args()
    c=json.loads(Path(a.config).read_text());c['_config_path']=a.config
    prepare(c,a.output)
