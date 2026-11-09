"""Remote CUDA only. GT never enters forward; default inference is disabled."""
import torch
from torch import nn
from torch.nn import functional as F
from prior_cuda_model import IndependentBoundaryResidual, cuda_only, boundary_target
from objectives_cuda import balanced_boundary_bce_logits
import plan as p


class BoundaryDelta(IndependentBoundaryResidual):
    def __init__(self, arm):
        assert arm in p.ARMS
        super().__init__()
        # Preserve constructor RNG, but remove the unbounded standalone GAP path.
        del self.classifier
        self.arm = arm
        self.delta_head = nn.Linear(64, 1, bias=False, device='cuda')
        nn.init.zeros_(self.delta_head.weight)

    def forward(self, rgb, baseline, enabled=False):
        cuda_only(baseline)
        if not enabled:
            return dict(logit=baseline, delta=torch.zeros_like(baseline), boundary_logit=None)
        cuda_only(rgb, *self.parameters())
        assert baseline.shape == (rgb.shape[0],) and bool(torch.isfinite(baseline).all())
        f0 = self.encoder[0](self.error_map(rgb))
        f1 = self.encoder[1](f0)
        f2 = self.encoder[2](f1)
        d1 = self.decoder[0](torch.cat((F.interpolate(f2, size=f1.shape[-2:], mode='bilinear', align_corners=False), f1), 1))
        d0 = self.decoder[1](torch.cat((F.interpolate(d1, size=f0.shape[-2:], mode='bilinear', align_corners=False), f0), 1))
        boundary = self.decoder[2](d0)
        contrast = (f2-F.avg_pool2d(f2, 5, 1, 2, count_include_pad=False)).abs()
        weight = F.interpolate(boundary.detach().sigmoid(), size=f2.shape[-2:], mode='area')
        cells = f2.shape[-2]*f2.shape[-1]
        floor = p.EPS if self.arm == p.ARMS[1] else .25
        mass = weight.sum((2,3))
        pooled = (contrast*weight).sum((2,3))/mass.clamp_min(cells*floor)
        delta = p.BOUND*self.delta_head(pooled).squeeze(1).tanh()
        return dict(logit=baseline.detach()+delta, delta=delta, boundary_logit=boundary,
                    weight_mass=mass.squeeze(1)/cells, pooled_norm=pooled.norm(dim=1))


def objective(out, labels, masks, valid, kernel):
    cuda_only(labels, masks, valid, kernel, out['logit'])
    image = F.binary_cross_entropy_with_logits(out['logit'], labels.float())
    if bool(valid.any()):
        aux = balanced_boundary_bce_logits(out['boundary_logit'][valid], boundary_target(masks[valid], kernel))
    else:
        aux = out['boundary_logit'].sum()*0
    penalty = out['delta'].square().mean()
    return image+.2*aux+p.PENALTY*penalty, image, aux, penalty
