import os
from pathlib import Path

def pixels(row,semantic=False):
    import numpy as np
    from PIL import Image
    path=row['image_path']
    with Path(path).open('rb') as f:
        with Image.open(f) as im:
            if semantic:
                # Exact historical preprocessing implementation, not a new crop.
                import gpu_pipeline as pipeline
                return pipeline.Pixels([row],False)[0]
            assert list(im.size)==row['source_size_wh']
            rgb=im.convert('RGB')
            if rgb.size!=(256,256):rgb=rgb.resize((256,256),Image.Resampling.BICUBIC)
            a=np.array(rgb,dtype=np.uint8).transpose(2,0,1).copy()
        if hasattr(os,'posix_fadvise'):os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    return a

def mask_pixels(row):
    import numpy as np
    from PIL import Image
    assert row['mask_valid'],'Missing masks never become auxiliary zeros'
    if not row['label']:
        assert row['method']=='real' and row['mask_path'] is None
        return np.zeros((256,256),dtype=np.uint8)
    assert row['mask_encoding']=='strict_equal_channels_binary_0_255'
    with Path(row['mask_path']).open('rb') as f:
        with Image.open(f) as im:
            assert im.size==(256,256);a=np.array(im)
            if a.ndim==3:assert a.shape[-1]==3 and bool((a==a[...,:1]).all());a=a[...,0]
            assert a.ndim==2 and bool(np.isin(np.unique(a),[0,255]).all())
        if hasattr(os,'posix_fadvise'):os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    return a
