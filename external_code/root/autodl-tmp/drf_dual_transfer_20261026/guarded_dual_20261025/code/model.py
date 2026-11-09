"""Frozen semantic anchor + bounded forensic correction; no performance claim."""
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

KERNEL_SHA = 'c53ad3a849fa64db1244152a3030c80a4d3b11fec0af81b4196d8e6dfb3db4b9'


@contextmanager
def initialization(name):
    seed = int.from_bytes(hashlib.sha256(f'{torch.initial_seed()}:{name}'.encode()).digest()[:8], 'big')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        yield


class SemanticHead(nn.Module):
    def __init__(self, dim=1024, hidden=256):
        super().__init__()
        with initialization('semantic'):
            self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(),
                                      nn.Dropout(.1), nn.Linear(hidden, 1))

    def forward(self, tokens):
        return self.head(tokens[:, 0].float()).squeeze(-1)


class ResidualFront(nn.Module):
    def __init__(self, width=32, adapt_scale=.1, use_filter=True):
        super().__init__()
        path = Path(__file__).parent / 'assets/teacher_SRM_Kernels.npy'
        if hashlib.sha256(path.read_bytes()).hexdigest() != KERNEL_SHA:
            raise ValueError('Kernel asset identity mismatch')
        a = np.load(path, allow_pickle=False)
        if a.shape != (30, 1, 5, 5) or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError('Invalid kernel bank')
        bank = torch.from_numpy(a.copy())
        bank = torch.cat([bank, bank[:2].rot90(1, (-2, -1))])
        bank = bank - bank.mean((-2, -1), keepdim=True)
        bank = bank / bank.abs().sum((-2, -1), keepdim=True).clamp_min(1e-8)
        self.register_buffer('base', bank)
        self.delta = nn.Parameter(torch.zeros_like(bank), requires_grad=bool(adapt_scale))
        self.adapt_scale, self.use_filter = float(adapt_scale), bool(use_filter)
        with initialization('residual_projection'):
            self.project = nn.Conv2d(96, width, 1, bias=False)

    def effective_kernel(self):
        d = self.delta - self.delta.mean((-2, -1), keepdim=True)
        d = d / d.abs().sum((-2, -1), keepdim=True).clamp_min(1.)
        return self.base + self.adapt_scale * d

    def forward(self, rgb):
        if self.use_filter:
            # Grouped outputs are [32 R responses, 32 G responses, 32 B responses].
            k = self.effective_kernel().to(rgb.dtype).repeat(3, 1, 1, 1)
            raw = F.conv2d(F.pad(rgb, (2, 2, 2, 2), mode='reflect'), k, groups=3)
            # Fixed scale; no label-dependent normalization or per-image stretching.
            raw = torch.tanh(raw / .1)
        else:
            # Same projection/encoder dimensions and common parameter initialization.
            raw = rgb.repeat_interleave(32, dim=1)
        return self.project(raw)


class GuardedDual(nn.Module):
    def __init__(self, semantic, spec, dim=1024):
        super().__init__()
        self.spec = dict(spec)
        self.semantic = semantic.requires_grad_(False).eval()
        h = int(spec['hidden'])
        with initialization('residual_encoder'):
            self.front = ResidualFront(adapt_scale=spec['adapt_scale'], use_filter=spec['use_filter'])
            self.encoder = nn.Sequential(
                nn.Conv2d(32, 32, 3, stride=2, padding=1, groups=32, bias=False),
                nn.Conv2d(32, h, 1, bias=False), nn.GroupNorm(8, h), nn.GELU(),
                nn.Conv2d(h, h, 3, stride=2, padding=1, groups=h, bias=False),
                nn.Conv2d(h, h, 1, bias=False), nn.GroupNorm(8, h), nn.GELU())
            self.r_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 1))
        with initialization('semantic_projection'):
            self.s_project = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, h))
        with initialization('interaction'):
            self.q_norm, self.kv_norm = nn.LayerNorm(h), nn.LayerNorm(h)
            self.cross = nn.MultiheadAttention(h, spec['heads'], batch_first=True)
        with initialization('correction'):
            self.correction = nn.Sequential(nn.LayerNorm(3 * h), nn.Linear(3 * h, h),
                                             nn.GELU(), nn.Linear(h, 1))
            nn.init.zeros_(self.correction[-1].weight)
            nn.init.zeros_(self.correction[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.semantic.eval()
        return self

    def forward(self, rgb, tokens):
        b = rgb.shape[0]
        has_s, has_r = self.spec['use_semantic'], self.spec['use_residual']
        if not has_s and not has_r:
            raise ValueError('Both branches disabled')
        with torch.no_grad():
            z_s = self.semantic(tokens) if has_s else rgb.new_zeros(b)
        if not has_r:
            return {'logit': z_s, 'semantic_logit': z_s, 'delta': torch.zeros_like(z_s)}
        grid = math.isqrt(tokens.shape[1] - 1)
        if grid * grid != tokens.shape[1] - 1:
            raise ValueError('Non-square semantic patch grid')
        r = self.encoder(self.front(rgb))
        r = F.adaptive_avg_pool2d(r, (grid, grid)).flatten(2).transpose(1, 2)
        z_r = self.r_head(r.mean(1)).squeeze(-1)
        s = self.s_project(tokens[:, 1:].float()) if has_s else r.new_zeros(r.shape)
        u = torch.zeros_like(r)
        if self.spec['use_interaction'] and has_s:
            yy, xx = torch.meshgrid(torch.arange(grid, device=r.device),
                                    torch.arange(grid, device=r.device), indexing='ij')
            coords = torch.stack([xx, yy], -1).reshape(-1, 2)
            mask = (coords[:, None] - coords[None, :]).abs().amax(-1) > self.spec['radius']
            memory = self.kv_norm(r)
            u, _ = self.cross(self.q_norm(s), memory, memory, attn_mask=mask, need_weights=False)
        raw = self.correction(torch.cat([s.mean(1), r.mean(1), u.mean(1)], -1)).squeeze(-1)
        delta = self.spec['correction_bound'] * torch.tanh(raw / self.spec['correction_bound'])
        return {'logit': z_s + delta, 'semantic_logit': z_s, 'residual_logit': z_r, 'delta': delta}


class ReliabilityGate(nn.Module):
    """Fitted after experts freeze, using source TRAIN families only."""
    def __init__(self, mode='sample'):
        super().__init__()
        if mode not in {'sample', 'global'}:
            raise ValueError('Unknown gate')
        self.mode = mode
        self.weight = nn.Parameter(torch.zeros(8))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, features):
        return torch.sigmoid(self.bias + (features * self.weight).sum(-1) if self.mode == 'sample'
                             else self.bias.expand(features.shape[0]))


def gate_features(s, d, r):
    """Per-video features. Label-free; supports any fixed frame count."""
    p, q = s.sigmoid(), r.sigmoid()
    eps = 1e-6
    entropy = -(p.clamp(eps, 1-eps).log() * p + (1-p).clamp(eps, 1-eps).log() * (1-p))
    return torch.stack([p.mean(-1), p.std(-1, unbiased=False), entropy.mean(-1),
                        d.mean(-1), d.abs().mean(-1), d.std(-1, unbiased=False),
                        q.mean(-1), (q-p).abs().mean(-1)], -1).detach()


def balanced_video_bce(probability, labels):
    """Equal class mass; preserve the deployment mean-frame probability."""
    if probability.ndim != 1 or labels.shape != probability.shape:
        raise ValueError('Expected one probability and label per video')
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError('Expected binary labels')
    if not ((labels == 0).any() and (labels == 1).any()):
        raise ValueError('Both classes required for balanced gate fitting')
    losses = F.binary_cross_entropy(probability.clamp(1e-6, 1-1e-6), labels, reduction='none')
    return .5 * (losses[labels == 0].mean() + losses[labels == 1].mean())
