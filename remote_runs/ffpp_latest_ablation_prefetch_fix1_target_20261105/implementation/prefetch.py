"""Bounded, ordered CPU-only decoding. No model, CUDA, RNG or dataset cache."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import os
from pathlib import Path


def ordered_batches(rows, decode, *, batch_size=16, parallel=True):
    """At most two pending batches plus the yielded batch; exceptions propagate.

    Consumers must close this generator on early exit (contextlib.closing).
    Workers never receive tensors or perform CUDA work. Input rows are not mutated.
    """
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    if not parallel:
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            yield start, chunk, decode(chunk)
        return
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='checked-pixels')
    pending = deque()
    starts = iter(range(0, len(rows), batch_size))

    def submit():
        start = next(starts, None)
        if start is not None:
            chunk = rows[start:start + batch_size]
            pending.append((start, chunk, pool.submit(decode, chunk)))

    try:
        submit()
        submit()
        while pending:
            start, chunk, future = pending.popleft()
            decoded = future.result()
            submit()
            yield start, chunk, decoded
            del decoded, future, chunk
    finally:
        for _, _, future in pending:
            future.cancel()
        # Compatible with pre-3.9 Python too: explicit cancellation above.
        pool.shutdown(wait=True)


def target_arrays(rows, pixel_hashes):
    """Identical original byte SHA / RGB / bicubic / uint8 CHW operations."""
    import numpy as np
    from PIL import Image
    arrays = []
    for row in rows:
        path = Path(row['image_path'])
        with path.open('rb') as stream:
            raw = stream.read()
            assert hashlib.sha256(raw).hexdigest() == pixel_hashes[str(path)]
            if hasattr(os, 'posix_fadvise'):
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        with Image.open(io.BytesIO(raw)) as image:
            rgb = image.convert('RGB')
            if rgb.size != (256, 256):
                rgb = rgb.resize((256, 256), Image.Resampling.BICUBIC)
            arrays.append(np.array(rgb, dtype=np.uint8).transpose(2, 0, 1).copy())
    return np.stack(arrays)
