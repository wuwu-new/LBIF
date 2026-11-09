"""CUDA-only full-source correction with frozen, compact semantic queries.

This is a separate conditional-transfer experiment, not fresh semantic training.
It deliberately reuses the immutable v1 training and evaluation machinery.
"""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

import gpu_pipeline as pipeline
import evaluate_frozen as evaluator


OriginalSemantic = pipeline.SemanticTokens
OriginalDual = pipeline.GuardedDual
original_encode = pipeline.encode
original_identities = pipeline.identities
original_configuration = pipeline.configuration
original_load_encoder = pipeline.load_encoder
original_gpu_audit = pipeline.gpu_audit


class CompactSemantic(OriginalSemantic):
    def forward(self, tokens):
        if tokens.shape[-1] == 192:
            return tokens[:, 0, 0]
        return super().forward(tokens)


class CompactEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.clip = pipeline.CLIPVisionModel.from_pretrained(
            config['clip_path'], local_files_only=True).cuda().eval().requires_grad_(False)
        self.anchor = OriginalSemantic(config['semantic_checkpoint']).cuda().eval()


@torch.no_grad()
def compact_encode(encoder, rgb):
    raw = original_encode(encoder.clip, rgb)
    queries = encoder.anchor.semantic_query(raw[:, 1:].float()).half().float()
    result = raw.new_zeros((len(raw), 257, 192))
    result[:, 1:] = queries
    result[:, 0, 0] = encoder.anchor(raw)
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite compact features')
    return result


class CompactDual(OriginalDual):
    def __init__(self, semantic, spec):
        super().__init__(semantic, spec, dim=192)


class CompactFeatures:
    """Keep patch features in fp16, but never round the semantic base logit."""
    def __init__(self, count):
        self.patches = torch.empty((count, 256, 192), dtype=torch.float16)
        self.logits = torch.empty(count, dtype=torch.float32)

    def __len__(self):
        return len(self.logits)

    def __getitem__(self, selection):
        patch = self.patches[selection].float()
        logit = self.logits[selection]
        result = patch.new_zeros((len(logit), 257, 192))
        result[:, 1:] = patch
        result[:, 0, 0] = logit
        return result


@torch.no_grad()
def compact_cache(encoder, rows, config, augmented=False):
    features = CompactFeatures(len(rows))
    pixels = torch.empty((len(rows), 3, 224, 224), dtype=torch.uint8)
    loader = DataLoader(pipeline.Pixels(rows, augmented), batch_size=config['batch_size'],
                        num_workers=config['workers'], pin_memory=True)
    offset = 0
    for rgb in loader:
        value = compact_encode(encoder, rgb).cpu()
        end = offset + len(rgb)
        features.patches[offset:end] = value[:, 1:].half()
        features.logits[offset:end] = value[:, 0, 0]
        pixels[offset:end] = (rgb * 255).round().to(torch.uint8)
        offset = end
        if offset % 2048 < len(rgb) or offset == len(rows):
            pipeline.log('COMPACT_RAM_CACHE', augmented=augmented, done=offset, total=len(rows))
    if offset != len(rows):
        raise ValueError('Incomplete compact cache')
    return {'tokens': features, 'rgb': pixels}


def configuration():
    config = original_configuration()
    config.update(name='guarded_full_source_compact_v2_20261026', epochs=10,
                  fit_families=10**9, pilot_val_families=10**9,
                  gate_families_per_fold=10**9, gate_steps=600,
                  semantic_query_dim=192,
                  semantic_query_source='frozen historical semantic_query; fp16 patches, fp32 logit',
                  protocol_scope='all prepared source families; inherited semantic seed2027; exploratory targets',
                  comparison_to_v1='larger source budget AND frozen compact query representation; not a one-factor ablation')
    return config


def identities(config):
    return {**original_identities(config), str(Path(__file__).resolve()): pipeline.digest(__file__),
            str(Path(evaluator.__file__).resolve()): pipeline.digest(evaluator.__file__)}


def memory_preflight(config):
    folder = pipeline.ROOT / 'guarded_dual_20261025/run_v1/manifests'
    counts = {name: len(json.loads((folder / (name + '.json')).read_text()))
              for name in ['fit', 'source_val', 'gate_fit', 'gate_holdout']}
    count = sum(counts.values()) + counts['fit']
    estimate = count * (256 * 192 * 2 + 4 + 3 * 224 * 224)
    limit = None
    for path in ['/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes']:
        p = Path(path)
        if p.exists() and p.read_text().strip().isdigit():
            candidate = int(p.read_text().strip())
            limit = min(limit or candidate, candidate)
    reserve = 6 * 1024**3
    if limit is not None and estimate + reserve > limit:
        raise RuntimeError('RAM budget insufficient; no eviction of existing files/jobs')
    pipeline.log('MEMORY_PREFLIGHT', counts=counts, estimated_cache_bytes=estimate,
                 reserve_bytes=reserve, container_limit_bytes=limit)


def gpu_audit(config, out):
    # Preserve the original raw-token legacy checkpoint audit without reinterpretation.
    pipeline.SemanticTokens = OriginalSemantic
    pipeline.GuardedDual = OriginalDual
    pipeline.load_encoder = original_load_encoder
    pipeline.encode = original_encode
    try:
        original_gpu_audit(config, out)
    finally:
        install_runtime()
    encoder = CompactEncoder(config)
    row = json.loads(Path(config['source_indexes']['val']).read_text())[0]
    rgb = pipeline.Pixels([row, row])[0].unsqueeze(0).repeat(2, 1, 1, 1)
    with torch.no_grad():
        raw = original_encode(encoder.clip, rgb)
        expected = encoder.anchor(raw)
        compact = compact_encode(encoder, rgb)
        torch.testing.assert_close(compact[:, 0, 0], expected, rtol=0, atol=0)
        saved = CompactFeatures(2)
        saved.patches[:] = compact[:, 1:].cpu().half()
        saved.logits[:] = compact[:, 0, 0].cpu()
        restored = saved[:].cuda()
        torch.testing.assert_close(restored, compact, rtol=0, atol=0)
        anchor = CompactSemantic(config['semantic_checkpoint']).cuda()
        torch.testing.assert_close(anchor(restored), expected, rtol=0, atol=0)
        model = CompactDual(anchor, config['model']).cuda().eval()
        torch.testing.assert_close(model(rgb.cuda(), restored)['logit'], expected, rtol=0, atol=0)
    pipeline.dump(out / 'COMPACT_AUDIT.json', {
        'semantic_logit_exact_raw_vs_compact': True,
        'cache_roundtrip_exact': True, 'zero_init_exit_exact': True,
        'query_dim': 192, 'patch_precision': 'fp16', 'base_logit_precision': 'fp32',
        'device': torch.cuda.get_device_name(), 'scope': 'contract checks, not performance evidence'})
    del encoder, raw, compact, model, anchor, restored, saved
    torch.cuda.empty_cache()


def install_runtime():
    pipeline.SemanticTokens = CompactSemantic
    pipeline.GuardedDual = CompactDual
    pipeline.load_encoder = CompactEncoder
    pipeline.encode = compact_encode
    pipeline.memory_cache = compact_cache
    pipeline.configuration = configuration
    pipeline.identities = identities
    pipeline.gpu_audit = gpu_audit
    evaluator.GuardedDual = CompactDual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['audit', 'train', 'source', 'targets'])
    parser.add_argument('output')
    args = parser.parse_args()
    install_runtime()
    if args.command in ['audit', 'train']:
        memory_preflight(configuration())
        sys.argv = [__file__, 'pilot' if args.command == 'train' else 'audit', args.output]
        pipeline.main()
    else:
        config = json.loads((Path(args.output) / 'CONFIG.json').read_text())
        if config.get('semantic_query_dim') != 192:
            raise ValueError('Not a compact v2 run')
        sys.argv = [__file__, args.command, args.output]
        evaluator.main()


if __name__ == '__main__':
    main()
