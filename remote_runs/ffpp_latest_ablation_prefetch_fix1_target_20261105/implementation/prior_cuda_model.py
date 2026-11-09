"""Independent residual expert. Remote CUDA only; no semantic or GT in forward."""
import torch
from torch import nn
from torch.nn import functional as F


def cuda_only(*xs):
    assert xs and all(x.is_cuda for x in xs),'CUDA only; no CPU model fallback'


def kernel(device):
    x=torch.arange(-2,3,device=device,dtype=torch.float32);v=torch.exp(-x.square()/2);v=v/v.sum()
    return (v[:,None]*v[None,:])[None,None]


def blur(x,k):
    cuda_only(x,k)
    return F.conv2d(F.pad(x.float(),(2,2,2,2),mode='reflect'),k.expand(x.shape[1],1,5,5),groups=x.shape[1])


def block(a,b,stride=1):
    return nn.Sequential(nn.Conv2d(a,b,3,stride=stride,padding=1,bias=False),nn.GroupNorm(8,b),nn.GELU(),
        nn.Conv2d(b,b,3,padding=1,bias=False),nn.GroupNorm(8,b),nn.GELU())


class IndependentBoundaryResidual(nn.Module):
    def __init__(self):
        super().__init__()
        assert torch.cuda.is_available()
        self.register_buffer('kernel',kernel(torch.device('cuda')))
        self.encoder=nn.ModuleList([block(3,16),block(16,32,2),block(32,64,2)])
        self.classifier=nn.Linear(64,1)
        self.decoder=nn.ModuleList([block(96,32),block(48,16),nn.Conv2d(16,1,1)])
        self.cuda()

    def error_map(self,rgb):
        cuda_only(rgb,self.kernel)
        assert rgb.shape[1:]==(3,256,256) and bool(torch.isfinite(rgb).all())
        assert bool(((rgb>=0)&(rgb<=1)).all())
        return rgb.float()-blur(rgb,self.kernel)

    def forward(self,rgb):
        cuda_only(rgb,*self.parameters())
        f0=self.encoder[0](self.error_map(rgb));f1=self.encoder[1](f0);f2=self.encoder[2](f1)
        logit=self.classifier(f2.mean((2,3))).squeeze(1)
        d1=self.decoder[0](torch.cat((F.interpolate(f2,size=f1.shape[-2:],mode='bilinear',align_corners=False),f1),1))
        d0=self.decoder[1](torch.cat((F.interpolate(d1,size=f0.shape[-2:],mode='bilinear',align_corners=False),f0),1))
        return dict(logit=logit,boundary_logit=self.decoder[2](d0))


@torch.no_grad()
def boundary_target(mask,k):
    cuda_only(mask,k);assert mask.shape[1:]==(1,256,256)
    assert bool(((mask==0)|(mask==1)).all())
    a=blur(mask,k).clamp(0,1);t=(4*a*(1-a)).clamp(0,1)
    return torch.where(t>1e-6,t,torch.zeros_like(t))


def balanced_boundary_mse(prediction,target):
    cuda_only(prediction,target);assert prediction.shape==target.shape
    e=(prediction.sigmoid()-target).square().flatten(1);pos=(target>1e-6).flatten(1);neg=~pos
    pc,nc=pos.sum(1),neg.sum(1)
    pe=(e*pos).sum(1)/pc.clamp_min(1);ne=(e*neg).sum(1)/nc.clamp_min(1)
    return ((pe+ne)/((pc>0).float()+(nc>0).float()).clamp_min(1)).mean()


def objectives(output,labels,masks,valid,k,arm):
    cuda_only(labels,masks,valid,k,*output.values())
    bce=F.binary_cross_entropy_with_logits(output['logit'],labels.float())
    # Invalid targets never enter the auxiliary objective, including normalization.
    if arm=='R1_Boundary' and bool(valid.any()):
        aux=balanced_boundary_mse(output['boundary_logit'][valid],boundary_target(masks[valid],k))
    else:aux=output['boundary_logit'].sum()*0 if arm=='R1_Boundary' else bce.new_zeros(())
    return bce+.2*aux,bce,aux
