"""Remote CUDA-only shallow/middle/deep residual routing, with matched control."""
import torch
from torch import nn
from torch.nn import functional as F
import multiscale_plan as plan


class ScaleAdapter(nn.Module):
    def __init__(self,body,arm,channels=128):
        super().__init__()
        plan.require(next(body.parameters()).is_cuda,'CUDA module required')
        plan.require(len(body)==7,'Pinned V5 inverted encoder geometry changed')
        self.body,self.arm=body,arm
        self.route=plan.route_indices(arm)
        self.normalize=nn.GroupNorm(8,channels,affine=False).cuda()
        # Do not perturb shared V5 initialization or subsequent training RNG.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            self.adapter=nn.Conv2d(3*channels,channels,1,bias=False,device='cuda')
            nn.init.zeros_(self.adapter.weight)
        self.audit_requested=False
        self.last_diagnostics=None

    def fuse(self,taps,grid=16):
        plan.require(all(value.is_cuda for value in taps),'CUDA activations required')
        deep=F.adaptive_avg_pool2d(taps[2],(grid,grid))
        features=[self.normalize(F.adaptive_avg_pool2d(taps[index],(grid,grid))) for index in self.route]
        addition=self.adapter(torch.cat(features,dim=1))
        if self.audit_requested:
            with torch.no_grad():
                self.last_diagnostics=dict(tap_shapes=[list(value.shape) for value in taps],
                    routed_indices=list(self.route),deep_rms=float(deep.square().mean().sqrt()),
                    addition_rms=float(addition.square().mean().sqrt()),
                    adapter_weight_rms=float(self.adapter.weight.square().mean().sqrt()))
            self.audit_requested=False
        return deep+addition

    def forward(self,value):
        taps=[]
        for index,module in enumerate(self.body):
            value=module(value)
            if index in (2,4,6):taps.append(value)
        return self.fuse(taps)


def upgrade(model,arm):
    plan.require(next(model.parameters()).is_cuda,'CUDA model required; no CPU fallback')
    plan.require(model.spec['encoder_kind']=='inverted' and model.spec['injection']=='layerscale','Wrong historical architecture')
    model.encoder=ScaleAdapter(model.encoder,arm,model.spec['hidden'])
    return model


def canonical_base_state(model):
    return {key.replace('encoder.body.','encoder.',1) if key.startswith('encoder.body.') else key:value
            for key,value in model.state_dict().items() if not key.startswith('encoder.adapter.')}


def route_probe(module):
    """Synthetic isolated activations verify actual shallow use, not performance."""
    saved=module.adapter.weight.detach().clone()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(182027);torch.cuda.manual_seed_all(182027)
        with torch.no_grad():nn.init.normal_(module.adapter.weight,std=.003)
        taps=[torch.randn((2,128,size,size),device='cuda',requires_grad=True) for size in (32,24,16)]
        output=module.fuse(taps)
        gradients=torch.autograd.grad(output.square().mean(),taps,allow_unused=True)
        masses=[float(gradient.abs().sum()) if gradient is not None else 0. for gradient in gradients]
        with torch.no_grad():
            altered=module.fuse([taps[0].roll(1,-1),taps[1],taps[2]])
            sensitivity=float((altered-output).abs().max())
            module.adapter.weight.copy_(saved)
    if module.arm==plan.ARMS[0]:
        plan.require(masses[0]==masses[1]==0 and sensitivity==0,'Deep-only control leaked shallow route')
    else:plan.require(all(value>0 for value in masses) and sensitivity>0,'Candidate ignores shallow or middle evidence')
    plan.require(bool((module.adapter.weight==0).all()),'Probe failed to restore zero initialization')
    return dict(activation_gradient_l1=masses,shallow_input_sensitivity=sensitivity,
                synthetic_activations_only=True,weights_restored=True)
