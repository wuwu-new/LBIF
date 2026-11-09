"""Remote regression checks only. Synthetic checks produce no AUC claim."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from model import SemanticHead, ResidualFront, GuardedDual, ReliabilityGate, gate_features, balanced_video_bce


class Regression(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2027);torch.set_num_threads(2)
        self.spec=json.loads((Path(__file__).parent/'config.remote.json').read_text())['model']
        self.spec['hidden']=16;self.spec['heads']=2

    def test_filters_stay_zero_dc_and_bounded(self):
        f=ResidualFront()
        with torch.no_grad():f.delta.normal_(0,10)
        k=f.effective_kernel();self.assertLess(float(k.sum((-2,-1)).abs().max()),1e-5)
        self.assertLessEqual(float((k-f.base).abs().sum((-2,-1)).max()),.100001)
        y=f(torch.ones(2,3,24,24)*.3);self.assertLess(float(y.abs().max()),1e-4)

    def test_grouped_rgb_order(self):
        from torch.nn import functional as F
        f=ResidualFront();rgb=torch.rand(2,3,24,24);k=f.effective_kernel()
        grouped=F.conv2d(F.pad(rgb,(2,2,2,2),mode='reflect'),k.repeat(3,1,1,1),groups=3)
        reference=torch.cat([F.conv2d(F.pad(rgb[:,i:i+1],(2,2,2,2),mode='reflect'),k) for i in range(3)],1)
        torch.testing.assert_close(grouped,reference,rtol=0,atol=0)

    def test_zero_initial_correction_and_frozen_anchor(self):
        s=SemanticHead(dim=8,hidden=16);m=GuardedDual(s,self.spec,dim=8).train()
        rgb=torch.rand(2,3,24,24);tokens=torch.rand(2,5,8)
        before={k:v.clone() for k,v in s.state_dict().items()}
        out=m(rgb,tokens);torch.testing.assert_close(out['logit'],s(tokens),rtol=0,atol=0)
        (out['logit'].sum()+out['residual_logit'].sum()).backward()
        self.assertFalse(s.training);self.assertTrue(all(p.grad is None for p in s.parameters()))
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in s.state_dict().items()))

    def test_no_semantic_cannot_read_semantic_tokens(self):
        spec={**self.spec,'use_semantic':False};m=GuardedDual(SemanticHead(8,16),spec,8).eval()
        with torch.no_grad():m.correction[-1].weight.normal_()
        rgb=torch.rand(2,3,24,24)
        torch.testing.assert_close(m(rgb,torch.rand(2,5,8))['logit'],m(rgb,torch.rand(2,5,8)*100)['logit'],rtol=0,atol=0)

    def test_no_residual_is_exact_semantic_exit(self):
        m=GuardedDual(SemanticHead(8,16),{**self.spec,'use_residual':False},8).eval()
        t=torch.rand(2,5,8)
        torch.testing.assert_close(m(torch.rand(2,3,24,24),t)['logit'],m.semantic(t),rtol=0,atol=0)

    def test_common_initialization_is_preserved(self):
        torch.manual_seed(2027);a=GuardedDual(SemanticHead(8,16),self.spec,8)
        torch.manual_seed(2027);b=GuardedDual(SemanticHead(8,16),{**self.spec,'use_filter':False},8)
        self.assertEqual(a.state_dict().keys(),b.state_dict().keys())
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)

    def test_bounded_correction_checkpoint_reload(self):
        a=GuardedDual(SemanticHead(8,16),self.spec,8).eval()
        with torch.no_grad():a.correction[-1].weight.normal_(0,100)
        b=GuardedDual(SemanticHead(8,16),self.spec,8).eval();b.load_state_dict(a.state_dict(),strict=True)
        rgb=torch.rand(2,3,24,24);t=torch.rand(2,5,8)
        out=a(rgb,t);self.assertLessEqual(float(out['delta'].abs().max()),2.)
        torch.testing.assert_close(out['logit'],b(rgb,t)['logit'],rtol=0,atol=0)

    def test_gate_features_and_constant_initialization(self):
        x=gate_features(torch.randn(3,32),torch.randn(3,32),torch.randn(3,32))
        g=ReliabilityGate('sample');torch.testing.assert_close(g(x),torch.ones(3)*.5,rtol=0,atol=0)
        self.assertTrue(torch.isfinite(x).all());self.assertEqual(tuple(x.shape),(3,8))

    def test_gate_loss_is_invariant_to_class_replication(self):
        p=torch.tensor([.2,.7],requires_grad=True);y=torch.tensor([0.,1.])
        reference=balanced_video_bce(p,y)
        repeated=balanced_video_bce(p[torch.tensor([0,1,1,1,1])],torch.tensor([0.,1.,1.,1.,1.]))
        torch.testing.assert_close(reference,repeated)
        repeated.backward();self.assertTrue(torch.isfinite(p.grad).all())
        with self.assertRaises(ValueError):balanced_video_bce(p,torch.zeros(2))


if __name__=='__main__':unittest.main(verbosity=2)
