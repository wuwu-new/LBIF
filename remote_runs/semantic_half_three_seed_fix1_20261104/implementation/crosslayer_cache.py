"""Remote CUDA feature construction; bounded RAM, disk additions for early layers only."""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import random
import crosslayer_plan as plan


@contextmanager
def isolated_rng():
    import numpy as np
    import torch
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def digest_read(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            value.update(block)
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return value.hexdigest()


class LayerTokens:
    def __init__(self, count):
        import torch
        self.patches = torch.empty((count, 256, 576), dtype=torch.float16)
        self.logits = torch.empty(count, dtype=torch.float32)

    def __getitem__(self, selection):
        patch, logit = self.patches[selection].float(), self.logits[selection]
        result = patch.new_zeros((len(logit), 257, 576))
        result[:, 1:] = patch
        result[:, 0, 0] = logit
        return result


class DiskLayers:
    def __init__(self, new_folder, old_folder, count):
        from disk_val_cache import DiskArray, DiskTokens
        self.early = DiskArray(new_folder / 'patches.bin', count, (256, 384), 'float16', 2)
        self.old = DiskTokens(old_folder, count)

    def __getitem__(self, selection):
        import torch
        old = self.old[selection]
        early = self.early[selection].float()
        result = old.new_zeros((len(old), 257, 576))
        result[:, 1:] = torch.cat([early, old[:, 1:]], dim=-1)
        result[:, 0, 0] = old[:, 0, 0]
        return result


def train_cache(encoder, rows, config, augmented, guard, resources, run):
    import torch
    from torch.utils.data import DataLoader
    import gpu_pipeline as pipeline
    from crosslayer_cuda import encode
    tokens = LayerTokens(len(rows))
    rgb_store = torch.empty((len(rows), 3, 224, 224), dtype=torch.uint8)
    offset = 0
    with isolated_rng():
        for rgb in DataLoader(pipeline.Pixels(rows, augmented), batch_size=16, num_workers=2, pin_memory=True):
            with torch.no_grad():
                value = encode(encoder, rgb).cpu()
            end = offset + len(rgb)
            tokens.patches[offset:end] = value[:, 1:].half()
            tokens.logits[offset:end] = value[:, 0, 0]
            rgb_store[offset:end] = (rgb * 255).round().to(torch.uint8)
            offset = end
            if offset % 2048 < len(rgb) or offset == len(rows):
                guard.check()
                resources(run)
                pipeline.log('CROSSLAYER_TRAIN_CACHE', augmented=augmented, done=offset, total=len(rows))
    plan.require(offset == len(rows), 'Incomplete TRAIN crosslayer cache')
    return dict(rgb=rgb_store, tokens=tokens)


def build_val(encoder, rows, config, run, old_folder, guard, resources, dump):
    import torch
    from torch.utils.data import DataLoader
    import gpu_pipeline as pipeline
    from crosslayer_cuda import encode, final_tokens
    from disk_val_cache import DiskArray, DiskTokens
    folder = run / 'VAL_EARLY'
    plan.require(not folder.exists(), 'Existing early-layer VAL cache preserved')
    folder.mkdir()
    old_rgb = DiskArray(old_folder / 'rgb.bin', len(rows), (3, 224, 224), 'uint8', 1)
    old_tokens = DiskTokens(old_folder, len(rows))
    offset = 0
    maximum = 0.
    with isolated_rng(), (folder / 'patches.bin').open('xb', buffering=0) as handle:
        for rgb in DataLoader(pipeline.Pixels(rows), batch_size=16, num_workers=2, pin_memory=True):
            with torch.no_grad():
                value = encode(encoder, rgb)
                final = final_tokens(value).cpu()
            end = offset + len(rgb)
            old = old_tokens[offset:end]
            plan.require(torch.equal((rgb * 255).round().to(torch.uint8), old_rgb[offset:end]), 'Old VAL RGB differs')
            plan.require(torch.equal(final, old), 'Old VAL final features/logits differ; do not relax precision')
            maximum = max(maximum, float((final - old).abs().max()))
            data = value[:, 1:, :384].half().cpu().contiguous().numpy().tobytes()
            position = handle.tell()
            view = memoryview(data)
            while view:
                wrote = handle.write(view)
                if not wrote:
                    raise OSError('Short early-layer cache write')
                view = view[wrote:]
            os.fdatasync(handle.fileno())
            os.posix_fadvise(handle.fileno(), position, len(data), os.POSIX_FADV_DONTNEED)
            offset = end
            if offset % 2048 < len(rgb) or offset == len(rows):
                guard.check()
                resources(run)
                pipeline.log('CROSSLAYER_VAL_CACHE', done=offset, total=len(rows), full_val_ram_cache=False)
    plan.require(offset == len(rows), 'Incomplete early VAL cache')
    record = dict(rows=offset, layers=[8, 16], projection='frozen historical semantic_query,fp16',
                  manifest_sha256=plan.fingerprint(rows), sha256=digest_read(folder / 'patches.bin'),
                  bytes=(folder / 'patches.bin').stat().st_size, max_final_feature_error=maximum,
                  old_VAL_RGB_final_logit_exact_all_frames=True, original_cache_duplicated=False)
    plan.require(record['bytes'] == plan.cache_estimate()['added_val_bytes'], 'VAL cache byte count changed')
    dump(folder / 'LAYOUT.json', record)
    return dict(rgb=old_rgb, tokens=DiskLayers(folder, old_folder, len(rows)))
