"""Independent CNN adaptation of mask-derived patch learning; remote CUDA only.

Reference: Cui et al., Forensics Adapter (CVPR 2025), supplied CC BY-NC 4.0
archive. The patch BCE head, balanced SupCon and CNN attachment are changes,
not a reproduction of their query-token/CLIP-interaction architecture.
"""
import torch
from torch import nn
from torch.nn import functional as F
from legacy_model_cuda import BoundaryDelta
from prior_cuda_model import cuda_only, boundary_target
from objectives_cuda import balanced_boundary_bce_logits
import patch_config as cfg


class PatchRegionDelta(BoundaryDelta):
    def __init__(self, variant):
        cfg.variant(variant)
        super().__init__('LegacyPoolDelta')
        self.patch_variant = variant
        # Allocate after the complete historical constructor: shared RNG draws
        # and shared initial weights are unchanged. Every variant has this head.
        self.patch_classifier = nn.Conv2d(64, 1, 1, device='cuda')

    def forward(self, rgb, baseline, enabled=False, return_patch=False):
        if not enabled:
            return super().forward(rgb, baseline, enabled=False)
        cuda_only(rgb, baseline, *self.parameters())
        assert baseline.shape == (len(rgb),) and bool(torch.isfinite(baseline).all())
        f0 = self.encoder[0](self.error_map(rgb))
        f1 = self.encoder[1](f0)
        f2 = self.encoder[2](f1)
        d1 = self.decoder[0](torch.cat((F.interpolate(f2, size=f1.shape[-2:], mode='bilinear', align_corners=False), f1), 1))
        d0 = self.decoder[1](torch.cat((F.interpolate(d1, size=f0.shape[-2:], mode='bilinear', align_corners=False), f0), 1))
        boundary = self.decoder[2](d0)
        contrast = (f2-F.avg_pool2d(f2, 5, 1, 2, count_include_pad=False)).abs()
        weight = F.interpolate(boundary.detach().sigmoid(), size=f2.shape[-2:], mode='area')
        cells = f2.shape[-2]*f2.shape[-1]
        mass = weight.sum((2, 3))
        pooled = (contrast*weight).sum((2, 3))/mass.clamp_min(cells*.25)
        delta = .5*self.delta_head(pooled).squeeze(1).tanh()
        out = dict(logit=baseline.detach()+delta, delta=delta, boundary_logit=boundary,
                   weight_mass=mass.squeeze(1)/cells, pooled_norm=pooled.norm(dim=1))
        if return_patch:
            features = F.adaptive_avg_pool2d(f2, (cfg.GRID_SIZE, cfg.GRID_SIZE))
            out['patch_logit'] = self.patch_classifier(features).flatten(1)
            out['patch_embedding'] = features.flatten(2).transpose(1, 2)
        return out


@torch.no_grad()
def patch_targets(masks, valid, image_labels):
    cuda_only(masks, valid, image_labels)
    n = len(image_labels)
    assert masks.shape == (n, 1, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
    assert valid.dtype == torch.bool and valid.shape == (n,)
    assert image_labels.shape == (n,) and bool(((image_labels == 0) | (image_labels == 1)).all())
    # Invalid bytes are never inspected or converted into supervisory zeros.
    selected = masks[valid]
    assert bool(torch.isfinite(selected).all())
    assert bool(((selected == 0) | (selected == 1)).all())
    assert not bool(selected[image_labels[valid] == 0].any()), 'Real-mask inconsistency'
    safe = torch.where(valid[:, None, None, None], masks.detach(), torch.zeros_like(masks))
    coverage = F.avg_pool2d(safe.float(), cfg.PATCH_SIZE, cfg.PATCH_SIZE).flatten(1)
    targets = (coverage > cfg.FAKE_COVERAGE_THRESHOLD).float()
    eligible = valid[:, None].expand_as(targets)
    return targets, eligible, coverage


def patch_classification_loss(logits, targets, eligible):
    cuda_only(logits, targets, eligible)
    assert logits.shape == targets.shape == eligible.shape
    assert eligible.dtype == torch.bool and bool(torch.isfinite(logits).all())
    if not bool(eligible.any()):
        return logits.sum()*0
    error = F.binary_cross_entropy_with_logits(logits.float(), targets.detach().float(), reduction='none')
    positive = eligible & (targets > .5)
    negative = eligible & ~positive
    pc, nc = positive.sum(1), negative.sum(1)
    value = ((error*positive).sum(1)/pc.clamp_min(1) +
             (error*negative).sum(1)/nc.clamp_min(1))
    classes = (pc > 0).float()+(nc > 0).float()
    return (value/classes.clamp_min(1))[classes > 0].mean()


def patch_contrastive_loss(embeddings, targets, eligible, image_labels):
    """Per-image, both-class SupCon. No diagonal or zero-masked fake pairs."""
    cuda_only(embeddings, targets, eligible, image_labels)
    assert embeddings.shape[:2] == targets.shape == eligible.shape
    assert eligible.dtype == torch.bool and bool(torch.isfinite(embeddings).all())
    losses = []
    for i in range(len(embeddings)):
        if int(image_labels[i]) != 1:
            continue
        labels = targets[i][eligible[i]].long()
        if int((labels == 0).sum()) < 2 or int((labels == 1).sum()) < 2:
            continue
        features = F.normalize(embeddings[i][eligible[i]].float(), dim=-1, eps=1e-6)
        scores = (features @ features.T)/cfg.TEMPERATURE
        off_diagonal = ~torch.eye(len(labels), dtype=torch.bool, device=scores.device)
        positive = (labels[:, None] == labels[None, :]) & off_diagonal
        # Select actual pairs before reduction; masked exp(0) never contributes.
        normalizer = scores.masked_fill(~off_diagonal, float('-inf')).logsumexp(1)
        log_probability = scores-normalizer[:, None]
        per_anchor = -log_probability.masked_fill(~positive, 0).sum(1)/positive.sum(1)
        losses.append(.5*(per_anchor[labels == 0].mean()+per_anchor[labels == 1].mean()))
    return torch.stack(losses).mean() if losses else embeddings.sum()*0


def objective(model, out, labels, masks, valid, *, split):
    if split != 'TRAIN':
        raise ValueError('Mask-supervised gradient objectives are TRAIN-only')
    cuda_only(labels, masks, valid, out['logit'], out['patch_logit'], out['patch_embedding'])
    assert labels.shape == out['logit'].shape
    targets, eligible, coverage = patch_targets(masks, valid, labels)
    assert out['patch_logit'].shape == (len(labels), cfg.GRID_SIZE**2)
    image = F.binary_cross_entropy_with_logits(out['logit'], labels.float())
    if bool(valid.any()):
        boundary = balanced_boundary_bce_logits(out['boundary_logit'][valid],
                                               boundary_target(masks[valid], model.kernel))
    else:
        boundary = out['boundary_logit'].sum()*0
    weights = cfg.variant(model.patch_variant)
    patch = patch_classification_loss(out['patch_logit'], targets, eligible)
    intra = patch_contrastive_loss(out['patch_embedding'], targets, eligible, labels)
    regularizer = out['delta'].square().mean()
    total = (image+cfg.BOUNDARY_WEIGHT*boundary+cfg.DELTA_PENALTY*regularizer+
             weights['patch_classification']*patch+weights['patch_contrast']*intra)
    return dict(total=total, image=image, boundary=boundary, regularizer=regularizer,
                patch_classification=patch, patch_contrast=intra,
                eligible_patches=eligible.sum().detach(),
                eligible_fake_patches=(eligible & (targets > .5)).sum().detach(),
                mixed_patches=(eligible & (coverage > 0) & (coverage < 1)).sum().detach())


def train_step(model, optimizer, rgb, baseline, labels, masks, valid, *, split):
    """One real optimizer integration point; caller owns budgets and provenance."""
    if split != 'TRAIN':
        raise ValueError('No VAL/TEST/target gradient updates')
    assert {id(p) for group in optimizer.param_groups for p in group['params']} == {
        id(p) for p in model.parameters()}
    model.train()
    optimizer.zero_grad(set_to_none=True)
    out = model(rgb, baseline, enabled=True, return_patch=True)
    terms = objective(model, out, labels, masks, valid, split=split)
    assert bool(torch.isfinite(terms['total']))
    terms['total'].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return {name: value.detach() for name, value in terms.items()}
