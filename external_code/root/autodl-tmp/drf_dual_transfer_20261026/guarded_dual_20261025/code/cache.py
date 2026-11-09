"""Identity-checked frozen CLIP caches. Run inference on the remote only."""
import hashlib
import json
import random
import io
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPVisionModel, CLIPImageProcessor
from prepare import digest, dump

MEAN = (.48145466, .4578275, .40821073)
STD = (.26862954, .26130258, .27577711)
PREPROCESSING = 'clip_shortest_bicubic_center224_rgb01_v1'


def source_identity(config):
    root = Path(__file__).parent
    h = hashlib.sha256()
    for p in sorted(root.glob('*.py')):
        h.update(p.name.encode());h.update(p.read_bytes())
    p=root/'assets/teacher_SRM_Kernels.npy';h.update(p.read_bytes())
    clip=Path(config['clip_path'])
    weights=next((clip/n for n in ['model.safetensors','pytorch_model.bin'] if (clip/n).is_file()),None)
    if weights is None:
        raise ValueError('Pinned local CLIP weights missing')
    return {'code_sha256':h.hexdigest(), 'clip_config_sha256':digest(clip/'config.json'),
            'clip_weights_sha256':digest(weights), 'preprocessing':PREPROCESSING}


class Pixels(Dataset):
    def __init__(self, rows, augmented=False):
        self.rows, self.augmented = rows, augmented
        self.processor = CLIPImageProcessor(do_resize=True, size={'shortest_edge':224},
            resample=Image.Resampling.BICUBIC, do_center_crop=True,
            crop_size={'height':224,'width':224}, do_rescale=True, do_normalize=False)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row=self.rows[i]
        if 'sha256' in row and digest(row['image_path']) != row['sha256']:
            raise ValueError('Source image changed after manifest audit')
        with Image.open(row['image_path']) as f:
            rgb=self.processor(f.convert('RGB'),return_tensors='pt')['pixel_values'][0]
        if self.augmented:
            # Fixed per-frame nuisance view. Seed is independent of class/method.
            rng=random.Random(int(hashlib.sha256(('nuisance-v1:'+row['image_path']).encode()).hexdigest()[:16],16))
            image=Image.fromarray((rgb.permute(1,2,0).numpy()*255).round().astype('uint8'))
            side=round(224*rng.uniform(.75,1.))
            image=image.resize((side,side),Image.Resampling.BICUBIC).resize((224,224),Image.Resampling.BICUBIC)
            image=image.filter(ImageFilter.GaussianBlur(rng.uniform(0,.4)))
            with io.BytesIO() as b:
                image.save(b,format='JPEG',quality=rng.randint(65,95));b.seek(0)
                with Image.open(b) as f:
                    a=np.asarray(f.convert('RGB')).copy()
            rgb=torch.from_numpy(a).permute(2,0,1).float()/255
        return rgb


class Cached(Dataset):
    def __init__(self, folder, rows, identity, semantic_only=False):
        self.folder=Path(folder);self.rows=rows;self.semantic_only=semantic_only
        self.meta=json.loads((self.folder/'COMPLETE.json').read_text())
        if self.meta['identity'] != identity or self.meta['rows_sha256'] != digest(self.folder.parent.parent/'manifests'/self.meta['manifest']):
            raise ValueError('Cache identity mismatch')
        if len(rows) != self.meta['count']:
            raise ValueError('Cache coverage mismatch')
        shape=(len(rows),*self.meta['token_shape'])
        self.tokens=np.memmap(self.folder/'tokens.f16',mode='r',dtype=np.float16,shape=shape)
        self.rgb=np.memmap(self.folder/'rgb.u8',mode='r',dtype=np.uint8,shape=(len(rows),3,224,224))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,i):
        if self.semantic_only:
            return {'tokens':torch.from_numpy(self.tokens[i,:1].copy()),
                    'label':float(self.rows[i]['label']), 'index':i}
        return {'tokens':torch.from_numpy(self.tokens[i].copy()),
                'rgb':torch.from_numpy(self.rgb[i].copy()).float()/255,
                'label':float(self.rows[i]['label']), 'index':i}


def build_caches(config, run):
    run=Path(run);identity=source_identity(config)
    path=run/'IDENTITY.json'
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError('Code/backbone changed since cache generation')
    else:dump(path,identity)
    # Deliberately offline; no model downloads or package changes.
    encoder=CLIPVisionModel.from_pretrained(config['clip_path'],local_files_only=True).eval().requires_grad_(False).cuda()
    mean=torch.tensor(MEAN,device='cuda')[None,:,None,None]
    std=torch.tensor(STD,device='cuda')[None,:,None,None]
    for part,aug in [('fit',False),('fit',True),('gate_fit',False),('gate_holdout',False),('source_val',False)]:
        folder=run/'cache'/(part+('_aug' if aug else ''))
        rows=json.loads((run/'manifests'/(part+'.json')).read_text())
        if (folder/'COMPLETE.json').exists():
            Cached(folder,rows,identity)
            print('CACHE_REUSE',str(folder),flush=True);continue
        folder.mkdir(parents=True,exist_ok=False)
        loader=DataLoader(Pixels(rows,aug),batch_size=config['cache_batch_size'],shuffle=False,
                          num_workers=config['workers'],pin_memory=True)
        pixels=np.memmap(folder/'rgb.u8',mode='w+',dtype=np.uint8,shape=(len(rows),3,224,224))
        tokens=None;offset=0
        with torch.inference_mode():
            for rgb in loader:
                n=len(rgb);pixels[offset:offset+n]=(rgb.numpy()*255).round().astype(np.uint8)
                x=rgb.cuda(non_blocking=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    out=encoder(pixel_values=(x-mean)/std).last_hidden_state
                if not torch.isfinite(out).all():raise ValueError('Nonfinite frozen tokens')
                a=out.float().cpu().numpy().astype(np.float16)
                if tokens is None:
                    token_shape=list(a.shape[1:])
                    tokens=np.memmap(folder/'tokens.f16',mode='w+',dtype=np.float16,shape=(len(rows),*token_shape))
                tokens[offset:offset+n]=a;offset+=n
                if offset%1024<len(rgb) or offset==len(rows):
                    print('CACHE',folder.name,offset,len(rows),flush=True)
        if offset!=len(rows):raise ValueError('Incomplete cache')
        pixels.flush();tokens.flush()
        dump(folder/'COMPLETE.json',{'identity':identity,'count':len(rows),'token_shape':token_shape,
            'manifest':part+'.json','rows_sha256':digest(run/'manifests'/(part+'.json')),
            'augmented':aug,'scope':'frozen CLIP float16 tokens; exact cached RGB shared by both branches'})
    del encoder
    torch.cuda.empty_cache()
