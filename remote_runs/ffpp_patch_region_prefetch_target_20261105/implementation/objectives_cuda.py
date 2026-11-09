"""Unregistered one-variable boundary-loss draft. Authorized remote CUDA only.

Reuse the frozen IndependentBoundaryResidual encoder, decoder and forward.
No semantic feedback, fusion, patch loss, extra input or training entry point.
Static tests do NOT certify CUDA equivalence, registration or efficacy.
"""
import torch
from torch.nn import functional as F


def balanced_boundary_bce_logits(prediction, target):
    """Same per-image support weighting as prior MSE; only replace the score.

    Keep soft targets and the original >1e-6 positive support definition.
    Do not binarize soft labels, add positive-class weights or apply sigmoid
    before BCEWithLogits. Caller excludes unavailable masks before this call.
    """
    assert prediction.is_cuda and target.is_cuda, 'Remote CUDA only'
    assert prediction.shape==target.shape and prediction.ndim==4
    assert prediction.shape[1:]==(1,256,256)
    assert bool(torch.isfinite(prediction).all()) and bool(torch.isfinite(target).all())
    assert bool(((target>=0)&(target<=1)).all())
    if len(prediction)==0:
        return prediction.sum()*0
    error=F.binary_cross_entropy_with_logits(prediction.float(),target.detach().float(),reduction='none').flatten(1)
    positive=(target>1e-6).flatten(1);negative=~positive
    pc,nc=positive.sum(1),negative.sum(1)
    pe=(error*positive).sum(1)/pc.clamp_min(1)
    ne=(error*negative).sum(1)/nc.clamp_min(1)
    return ((pe+ne)/((pc>0).float()+(nc>0).float()).clamp_min(1)).mean()


def objectives(output, labels, masks, valid, kernel, arm, boundary_target, previous_mse):
    """Loss-only interface; dependencies must be SHA-locked by a future runner.

    MSE_control and BCE_logits share all other operations and coefficient .2.
    NoAux is for synthetic checks / potential separately budgeted control;
    it does not authorize another training arm.
    """
    if arm not in ('MSE_control','BCE_logits','NoAux'):
        raise ValueError('Unknown draft arm')
    assert all(x.is_cuda for x in (labels,masks,valid,kernel,*output.values())), 'Remote CUDA only'
    assert valid.dtype==torch.bool and valid.shape==labels.shape
    assert output['logit'].shape==labels.shape and masks.shape==(len(labels),1,256,256)
    bce=F.binary_cross_entropy_with_logits(output['logit'],labels.float())
    if arm=='NoAux':
        return bce,bce,bce.new_zeros(())
    if bool(valid.any()):
        truth=boundary_target(masks[valid],kernel)
        prediction=output['boundary_logit'][valid]
        aux=previous_mse(prediction,truth) if arm=='MSE_control' else balanced_boundary_bce_logits(prediction,truth)
    else:
        # All unavailable: no mask pixels read, no fabricated zero target.
        aux=output['boundary_logit'].sum()*0
    return bce+.2*aux,bce,aux
