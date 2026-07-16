"""
RANGER Phase-2 — QAT building blocks.

Implements the training-time mechanisms the theory says are required below
~3 bits (ParetoQ 2502.02631) and the RANGER-specific constraints discovered in
the mock runs:

  * FakeQuantLinear   — STE fake-quant of nn.Linear weights.
                        int-b (per-out-channel symmetric) or ternary (BitNet
                        b1.58 recipe: s = mean|W|, levels {-1,0,+1}).
  * super-weight split (constraint G, E5) — the top-p%% |W| entries stay in a
                        full-precision sparse path; ONLY the outlier-free dense
                        remainder is quantized (and its scale is computed
                        without the outliers, so they can't inflate it).
  * nested precision  (Pillar 2) — per-out-channel importance ordering; the
                        important half gets bits+1, the rest bits-1 (equal avg).
  * kurtosis_loss     (Pillar 1 / S2D 2602.14432, KurTail 2503.01483) — a
                        differentiable hinge on hidden-state excess kurtosis;
                        trains the residual stream to be incoherence-native.
  * optional exact online Hadamard on the input dim (QuaRot R4-style): rotates
                        x and W by the same involutive H so the product is
                        mathematically unchanged while both operands quantize
                        in the incoherent basis.

Everything is CPU-safe. No custom CUDA kernels required.
"""
import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase1"))
from common import hadamard, next_pow2  # noqa: E402


def ste_round(x):
    """round() with a straight-through gradient."""
    return x + (x.round() - x).detach()


def _fake_quant(W, bits, ternary=False):
    """Symmetric per-out-channel fake quant with STE. W: [out, in].

    bits==1 is true binary (sign * mean|W|, BitNet-style) — the symmetric int
    formula degenerates there (qmax would be 0), so it gets its own path.
    """
    if ternary:
        s = W.abs().mean(dim=1, keepdim=True).detach().clamp_min(1e-8)  # BitNet b1.58
        return torch.clamp(ste_round(W / s), -1, 1) * s
    if int(bits) <= 1:                                   # binary: {-s, +s}
        s = W.abs().mean(dim=1, keepdim=True).detach().clamp_min(1e-8)
        sign = torch.where(W >= 0, 1.0, -1.0)
        return W + (sign * s - W).detach()               # STE
    qmax = 2 ** (int(bits) - 1) - 1
    s = (W.abs().amax(dim=1, keepdim=True).detach().clamp_min(1e-8)) / qmax
    return torch.clamp(ste_round(W / s), -qmax - 1, qmax) * s


class FakeQuantLinear(nn.Module):
    """Drop-in nn.Linear replacement with STE weight fake-quant.

    Options (all composable):
      bits            target weight bits (ignored if ternary=True)
      ternary         BitNet b1.58 levels {-1,0,+1}
      sw_mask         bool [out,in]: True entries stay full-precision (sparse
                      path, split BEFORE quantization — constraint G)
      hi_mask         bool [out]: channels quantized at bits+1; the rest at
                      bits-1 (nested precision at equal average bits)
      rotate          exact online Hadamard on the in-dim (pads to pow2)
    """

    def __init__(self, linear: nn.Linear, bits=3, ternary=False,
                 sw_mask=None, hi_mask=None, rotate=False):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = nn.Parameter(linear.weight.data.clone())
        self.bias = (nn.Parameter(linear.bias.data.clone())
                     if linear.bias is not None else None)
        self.bits, self.ternary, self.rotate = bits, ternary, rotate
        self.register_buffer("sw_mask",
                             sw_mask if sw_mask is not None
                             else torch.zeros_like(self.weight, dtype=torch.bool))
        self.hi_mask = None
        if hi_mask is not None:
            self.register_buffer("_hi", hi_mask.view(-1, 1))
            self.hi_mask = True

    def effective_weight(self):
        W = self.weight
        # constraint G: zero the super-weights out of the dense part so they
        # can neither be quantized nor inflate the scale
        W_dense = W.masked_fill(self.sw_mask, 0.0)
        if self.hi_mask is not None and not self.ternary:
            Wq_hi = _fake_quant(W_dense, self.bits + 1)
            Wq_lo = _fake_quant(W_dense, max(self.bits - 1, 1))
            Wq = torch.where(self._hi, Wq_hi, Wq_lo)
        else:
            Wq = _fake_quant(W_dense, self.bits, ternary=self.ternary)
        # sparse full-precision path (still trainable)
        return Wq + W * self.sw_mask

    def forward(self, x):
        Weff = self.effective_weight()
        if self.rotate:
            n = self.in_features
            m = next_pow2(n)
            xr = hadamard(F.pad(x, (0, m - n)) if m != n else x)
            Wr = hadamard(F.pad(Weff, (0, m - n)) if m != n else Weff)
            return F.linear(xr, Wr, self.bias)   # exact: xH·(WH)^T = x·W^T
        return F.linear(x, Weff, self.bias)

    def extra_repr(self):
        mode = "ternary" if self.ternary else f"{self.bits}b"
        nest = "+nested" if self.hi_mask is not None else ""
        sw = f"+sw({int(self.sw_mask.sum())})" if bool(self.sw_mask.any()) else ""
        rot = "+rot" if self.rotate else ""
        return f"in={self.in_features}, out={self.out_features}, {mode}{nest}{sw}{rot}"


@torch.no_grad()
def _superweight_mask(W, pct):
    """Bool mask of the top-pct fraction of |W| entries (the super-weights)."""
    k = max(1, int(pct * W.numel()))
    thr = torch.kthvalue(W.abs().flatten(), W.numel() - k + 1).values
    return W.abs() >= thr


def convert_to_qat(model, bits=3, ternary=False, superweight_pct=0.0,
                   nested=False, protect_downproj=False, rotate_downproj=False,
                   skip=("lm_head",)):
    """Swap every nn.Linear (except `skip`) for a FakeQuantLinear.

    nested            -> per-out-channel importance split (bits±1, equal avg)
    protect_downproj  -> down_proj gets bits+1 (QAT scaling law: FC2 dominates)
    rotate_downproj   -> exact online Hadamard on down_proj inputs (R4-style)
    Returns the number of layers converted.
    """
    n_conv = 0
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            full = f"{parent_name}.{child_name}" if parent_name else child_name
            if not isinstance(child, nn.Linear) or any(s in full for s in skip):
                continue
            W = child.weight.data
            sw = _superweight_mask(W, superweight_pct) if superweight_pct > 0 else None
            hi = None
            if nested and not ternary:
                imp = W.abs().sum(dim=1)
                hi = torch.zeros(W.shape[0], dtype=torch.bool, device=W.device)
                hi[torch.argsort(imp, descending=True)[: W.shape[0] // 2]] = True
            b = bits + 1 if (protect_downproj and "down_proj" in full) else bits
            rot = rotate_downproj and "down_proj" in full
            parent._modules[child_name] = FakeQuantLinear(
                child, bits=b, ternary=ternary, sw_mask=sw, hi_mask=hi, rotate=rot)
            n_conv += 1
    return n_conv


def kurtosis_loss(hidden_states, tau=1.0):
    """Differentiable hinge on excess kurtosis of the residual stream.

    hidden_states: tuple of [B, T, D] tensors (from output_hidden_states=True).
    Penalty is log1p(relu(kurt - tau)) per layer — log-scaled because raw excess
    kurtosis reaches the thousands at the massive-emergence layer and a linear
    penalty would swamp the LM loss. tau=1.0 leaves near-Gaussian layers alone.
    """
    total = hidden_states[0].new_zeros(())
    for h in hidden_states[1:]:                     # skip the embedding layer
        x = h.float().reshape(-1)
        mu = x.mean()
        sd = x.std().clamp_min(1e-6)
        kurt = (((x - mu) / sd) ** 4).mean() - 3.0
        total = total + torch.log1p(F.relu(kurt - tau))
    return total / max(len(hidden_states) - 1, 1)


def average_bits(model):
    """Report the effective average weight bits across converted layers."""
    tot_w, tot_b = 0, 0.0
    for m in model.modules():
        if isinstance(m, FakeQuantLinear):
            n = m.weight.numel()
            if m.ternary:
                b = 1.58
            elif m.hi_mask is not None:
                b = ((m._hi.sum() * (m.bits + 1)
                      + (~m._hi).sum() * max(m.bits - 1, 1)).float()
                     / m._hi.numel()).item()
            else:
                b = float(m.bits)
            n_sw = int(m.sw_mask.sum())
            tot_b += b * (n - n_sw) + 16.0 * n_sw   # sparse path stored fp16
            tot_w += n
    return tot_b / max(tot_w, 1)
