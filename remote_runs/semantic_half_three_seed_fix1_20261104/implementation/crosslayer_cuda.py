"""Remote CUDA-only crosslayer encoder/readout and independently trained no-image control."""
import torch
from torch import nn
from torch.nn import functional as F
import crosslayer_plan as plan


@torch.no_grad()
def encode(encoder, rgb):
    """Fresh real hidden states, not reconstructed from old192D final-layer cache."""
    import gpu_pipeline as pipeline
    x = rgb.cuda(non_blocking=True)
    mean, std = [x.new_tensor(values)[None, :, None, None] for values in (pipeline.MEAN, pipeline.STD)]
    with torch.autocast('cuda', dtype=torch.float16):
        out = encoder.clip(pixel_values=(x - mean) / std, output_hidden_states=True, return_dict=True)
    plan.require(len(out.hidden_states) == 25 and out.last_hidden_state.shape[1:] == (257, 1024), 'Pinned CLIP geometry changed')
    plan.require(torch.equal(out.hidden_states[24], out.last_hidden_state), 'Final hidden-state indexing differs')
    raw = out.last_hidden_state.half().float()
    # Preserve the original257-token batch stride entering Linear. Slice-then-cast
    # materializes256 contiguous tokens and changes the CUDA Linear numerical path.
    projected = []
    for layer in plan.LAYERS:
        layer_raw = out.hidden_states[layer].half().float()
        projected.append(encoder.anchor.semantic_query(layer_raw[:, 1:].float()).half().float())
    value = raw.new_zeros((len(raw), 257, 576))
    value[:, 1:] = torch.cat(projected, dim=-1)
    value[:, 0, 0] = encoder.anchor(raw)
    plan.require(bool(torch.isfinite(value).all()), 'Nonfinite crosslayer features')
    return value


def final_tokens(tokens):
    plan.require(tokens.is_cuda and tokens.shape[1:] == (257, 576), 'CUDA crosslayer tokens required')
    result = tokens.new_zeros((len(tokens), 257, 192))
    result[:, 1:] = tokens[:, 1:, 384:]
    result[:, 0, 0] = tokens[:, 0, 0]
    return result


class LayerReadout(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.route = plan.routes(arm)
        self.normalize = nn.LayerNorm(192, elementwise_affine=False, device='cuda')
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            self.project = nn.Linear(576, 192, bias=False, device='cuda')
            with torch.no_grad():
                self.project.weight.copy_(torch.cat([torch.eye(192, device='cuda') / 3] * 3, dim=1))

    def forward(self, tokens):
        layers = tokens[:, 1:].reshape(len(tokens), 256, 3, 192)
        selected = torch.cat([self.normalize(layers[:, :, index]) for index in self.route], dim=-1)
        patch = self.project(selected)
        result = patch.new_zeros((len(patch), 257, 192))
        result[:, 1:] = patch
        result[:, 0, 0] = tokens[:, 0, 0]
        return result


class CrossLayerDual(nn.Module):
    def __init__(self, base, arm):
        super().__init__()
        plan.require(next(base.parameters()).is_cuda and arm in plan.ARMS, 'CUDA model and known arm required')
        self.base, self.arm = base, arm
        self.readout = LayerReadout(arm)
        if arm == 'CrossLayerNoImage':
            # Independent token-only branch, not Full with a zeroed input or frozen weights.
            # Retain width/depth and ALL blocks; avoid erasing token-grid detail by downsampling.
            self.base.front = nn.Identity()
            self.base.encoder.body[3] = nn.Identity()
            self.base.encoder.body[5] = nn.Identity()
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(2202027)
                self.token_input = nn.Linear(192, 32, bias=False, device='cuda')

    def forward(self, rgb, tokens):
        plan.require(rgb.is_cuda and tokens.is_cuda, 'No CPU model fallback')
        compact = self.readout(tokens)
        if self.arm == 'CrossLayerNoImage':
            # rgb is only an API placeholder: no values, summaries, shape-derived geometry or SRM are read.
            image = self.token_input(compact[:, 1:]).transpose(1, 2).reshape(len(tokens), 32, 16, 16)
        else:
            image = rgb
        return self.base(image, compact)


def route_probe(model, rgb, tokens):
    """Label-free actual CUDA probes; no optimizer update or retained weight change."""
    with torch.no_grad():
        baseline = model.readout(tokens)
        altered = tokens.clone()
        altered[:, 1:, :192] = altered[:, 1:, :192].roll(1, dims=1)
        middle_error = float((model.readout(altered) - baseline).abs().max())
        if model.arm == 'LastLayerResidual':
            plan.require(middle_error == 0, 'Last-layer control leaked intermediate features')
        else:
            plan.require(middle_error > 0, 'Crosslayer readout ignores real intermediate features')
    x = rgb.detach().clone().requires_grad_(True)
    result = model(x, tokens)
    gradient = torch.autograd.grad(result['residual_logit'].sum(), x, allow_unused=True)[0]
    if model.arm == 'CrossLayerNoImage':
        plan.require(gradient is None, 'No-image control leaks RGB gradient')
        with torch.no_grad():
            changed = model(torch.flip(rgb, dims=[-1]), tokens)
        plan.require(torch.equal(result['residual_logit'], changed['residual_logit']) and
                     torch.equal(result['logit'], changed['logit']), 'No-image control leaks RGB values')
    else:
        plan.require(gradient is not None and bool(gradient.abs().sum() > 0), 'Full model ignores image residual')
    return dict(intermediate_readout_sensitivity=middle_error,
                image_auxiliary_gradient_l1=float(gradient.abs().sum()) if gradient is not None else 0.,
                image_independent=model.arm == 'CrossLayerNoImage', optimizer_updates=0)
