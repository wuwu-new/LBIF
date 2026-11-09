"""Remote input decoding: identical legacy pixels, SHA checked on every read."""
import hashlib
import io
import os
from pathlib import Path


def checked_bytes(path, expected):
    with Path(path).open('rb') as handle:
        raw = handle.read()
        if hasattr(os, 'posix_fadvise'):
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    assert hashlib.sha256(raw).hexdigest() == expected, ('Input changed', path)
    return raw


def pixels(row):
    import numpy as np
    from PIL import Image
    raw = checked_bytes(row['image_path'], row['sha256'])
    with Image.open(io.BytesIO(raw)) as image:
        assert list(image.size) == row['source_size_wh']
        rgb = image.convert('RGB')
        if rgb.size != (256, 256):
            rgb = rgb.resize((256, 256), Image.Resampling.BICUBIC)
        return np.array(rgb, dtype=np.uint8).transpose(2, 0, 1).copy()


def mask_pixels(row):
    import numpy as np
    from PIL import Image
    assert row['mask_valid'], 'Missing masks never become auxiliary zeros'
    if not row['label']:
        assert row['method'] == 'real' and row['mask_path'] is None
        return np.zeros((256, 256), dtype=np.uint8)
    assert row['mask_encoding'] == 'strict_equal_channels_binary_0_255'
    raw = checked_bytes(row['mask_path'], row['mask_sha256'])
    with Image.open(io.BytesIO(raw)) as image:
        assert image.size == (256, 256)
        array = np.array(image)
        if array.ndim == 3:
            assert array.shape[-1] == 3 and bool((array == array[..., :1]).all())
            array = array[..., 0]
        assert array.ndim == 2 and bool(np.isin(np.unique(array), [0, 255]).all())
        return array
