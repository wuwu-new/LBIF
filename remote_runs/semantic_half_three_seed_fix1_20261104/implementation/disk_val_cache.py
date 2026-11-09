"""Compact VAL disk transport. Imports torch only inside the remote CUDA runtime."""
import json
import os
from pathlib import Path


def bounds(selection,count):
    if not isinstance(selection,slice) or selection.step not in (None,1):raise ValueError('Only contiguous slices allowed')
    start,stop,step=selection.indices(count)
    if stop<=start:raise ValueError('Empty disk cache slice')
    return start,stop


class DiskArray:
    def __init__(self,path,count,shape,dtype,itemsize):
        self.path,self.count,self.shape,self.dtype=Path(path),count,shape,dtype
        self.row_bytes=itemsize
        for dim in shape:self.row_bytes*=dim
        if self.path.stat().st_size!=count*self.row_bytes:raise ValueError('Wrong disk cache byte count')

    def __getitem__(self,selection):
        import torch
        start,stop=bounds(selection,self.count);needed=(stop-start)*self.row_bytes
        fd=os.open(self.path,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
        try:
            data=bytearray();offset=start*self.row_bytes
            while len(data)<needed:
                block=os.pread(fd,needed-len(data),offset+len(data))
                if not block:raise ValueError('Truncated disk cache')
                data.extend(block)
            os.posix_fadvise(fd,offset,needed,os.POSIX_FADV_DONTNEED)
        finally:os.close(fd)
        return torch.frombuffer(data,dtype=getattr(torch,self.dtype)).reshape(stop-start,*self.shape)


class DiskTokens:
    def __init__(self,folder,count):
        self.patches=DiskArray(folder/'patches.bin',count,(256,192),'float16',2)
        self.logits=DiskArray(folder/'logits.bin',count,(),'float32',4)

    def __getitem__(self,selection):
        patch=self.patches[selection].float();logit=self.logits[selection]
        result=patch.new_zeros((len(logit),257,192));result[:,1:]=patch;result[:,0,0]=logit
        return result


def create_cache(encoder,rows,config,folder,compact,pipeline,guard,resource_check):
    import torch
    from torch.utils.data import DataLoader
    folder=Path(folder);folder.mkdir();paths={key:folder/(key+'.bin') for key in ('rgb','patches','logits')}
    handles={key:path.open('xb',buffering=0) for key,path in paths.items()}
    offset=0
    try:
        loader=DataLoader(pipeline.Pixels(rows,False),batch_size=16,num_workers=config['workers'],pin_memory=True)
        for rgb in loader:
            with torch.no_grad():tokens=compact.compact_encode(encoder,rgb).cpu()
            parts={'rgb':(rgb*255).round().to(torch.uint8),'patches':tokens[:,1:].half(),'logits':tokens[:,0,0].float()}
            for key,tensor in parts.items():
                handle=handles[key];position=handle.tell();data=tensor.contiguous().numpy().tobytes();view=memoryview(data)
                while view:
                    wrote=handle.write(view)
                    if not wrote:raise OSError('Short disk cache write')
                    view=view[wrote:]
                os.fdatasync(handle.fileno());os.posix_fadvise(handle.fileno(),position,len(data),os.POSIX_FADV_DONTNEED)
            offset+=len(rgb)
            if offset%2048<len(rgb) or offset==len(rows):guard.check();resource_check();pipeline.log('COMPACT_RAM_CACHE',done=offset,total=len(rows),storage='disk',augmented=False)
    finally:
        for handle in handles.values():handle.close()
    if offset!=len(rows):raise ValueError('Incomplete disk VAL cache')
    with (folder/'LAYOUT.json').open('x',encoding='utf-8') as handle:json.dump(dict(count=offset,rgb='uint8',patches='fp16',semantic_logit='fp32',full_val_ram_cache=False,scoped_cache_advice=True),handle)
    return {'rgb':DiskArray(paths['rgb'],offset,(3,224,224),'uint8',1),'tokens':DiskTokens(folder,offset)}
