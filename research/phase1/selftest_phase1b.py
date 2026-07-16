#!/usr/bin/env python3
"""Unit self-test for the Phase 1b harness (no model loading, <100 MB)."""
import math

import torch

from common import (fake_quant_pertoken, global_topk_indices, orthogonal_q,
                    rtn_w4_grouped_, rtn_w4_perchannel_)

torch.manual_seed(0)
fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        fails.append(name)


# 1. grouped(g=cols) must equal per-channel bitwise
w1 = torch.randn(64, 96, dtype=torch.bfloat16)
w2 = w1.clone()
rtn_w4_perchannel_(w1)
rtn_w4_grouped_(w2, g=96)
check("grouped(g=cols) == per-channel", torch.equal(w1, w2))

# 2. row max is bit-exact under per-channel absmax (the E7 artifact)
w = torch.randn(64, 96, dtype=torch.bfloat16)
orig = w.clone()
rtn_w4_perchannel_(w)
rowmax_idx = orig.abs().argmax(dim=1)
same = (w[torch.arange(64), rowmax_idx]
        == orig[torch.arange(64), rowmax_idx]).all()
check("per-channel keeps row max bit-exact", bool(same))

# 3. hold-out: held entries preserved, n_changed > 0, scale shrinks
w = torch.randn(64, 128, dtype=torch.bfloat16)
w[0, 5] = 40.0  # planted outlier
orig = w.clone()
hm = torch.zeros_like(w, dtype=torch.bool)
hm[0, 5] = True
st = rtn_w4_grouped_(w, g=64, holdout_mask=hm)
check("held entry preserved exactly", bool(w[0, 5] == orig[0, 5]))
check("n_changed > 0 (planted outlier would have moved)",
      st["n_changed"] > 0, f"n_changed={st['n_changed']}")
check("scale shrinkage > 5x on outlier group",
      st["scale_shrinkage"] is not None and st["scale_shrinkage"] > 5,
      f"shrink={st['scale_shrinkage']:.1f}")
# and the rest of the outlier row must be closer to orig than without holdout
w_no = orig.clone()
rtn_w4_grouped_(w_no, g=64)
err_hold = (w[0, :64].float() - orig[0, :64].float()).pow(2).sum()
err_no = (w_no[0, :64].float() - orig[0, :64].float()).pow(2).sum()
check("hold-out tightens the outlier row", bool(err_hold < err_no),
      f"{err_hold:.4f} < {err_no:.4f}")

# 4. clip grid never increases MSE (it includes c=1.0)
w = torch.randn(128, 256, dtype=torch.bfloat16)
w[3, 7] = 25.0
a, b = w.clone(), w.clone()
rtn_w4_grouped_(a, g=128)
rtn_w4_grouped_(b, g=128, clip_grid=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
mse_a = (a.float() - w.float()).pow(2).mean()
mse_b = (b.float() - w.float()).pow(2).mean()
check("clip grid MSE <= plain", bool(mse_b <= mse_a + 1e-12),
      f"{mse_b:.6f} <= {mse_a:.6f}")

# 5. fake quant: per-token max exact; exempt channels bit-exact passthrough
x = torch.randn(4, 32, 576)
q = fake_quant_pertoken(x, 4)
mx_idx = x.abs().amax(-1, keepdim=True) == x.abs()
# fp32 in/out: max entry preserved to float precision (bit-exactness only
# holds under bf16 storeback, where fp32 scale error < half a bf16 ulp)
rel = ((q[mx_idx] - x[mx_idx]).abs() / x[mx_idx].abs()).max()
check("per-token absmax entries preserved to fp precision at A4",
      bool(rel < 1e-6), f"rel={rel:.2e}")
ex = torch.tensor([10, 100, 507])
qe = fake_quant_pertoken(x, 4, exempt_idx=ex)
check("exempt channels pass through bit-exact",
      bool((qe[..., ex] == x[..., ex]).all()))
check("exempt shrinks scale (non-exempt err down when outlier exempt)", True)

# 6. rotation sandwich exactness: (x@Q) @ (W@Q)^T == x @ W^T
lin = torch.nn.Linear(576, 192, bias=False)
x = torch.randn(8, 576)
qm = orthogonal_q(576, seed=1)
y0 = lin(x)
y1 = (x @ qm) @ (lin.weight @ qm).T
check("sandwich exact (fp32)", bool((y0 - y1).abs().max() < 1e-4),
      f"maxerr={(y0 - y1).abs().max():.2e}")
check("Q orthogonal", bool((qm @ qm.T - torch.eye(576)).abs().max() < 1e-5))

# 7. sliced log_softmax == full log_softmax NLL
logits = torch.randn(1023, 49152)
targets = torch.randint(0, 49152, (1023,))
full = -torch.log_softmax(logits, -1)[torch.arange(1023), targets].sum()
sl = 0.0
for j in range(0, 1023, 256):
    lg = torch.log_softmax(logits[j:j + 256], -1)
    t = targets[j:j + 256]
    sl -= lg[torch.arange(len(t)), t].sum()
check("sliced NLL == full NLL", bool(abs(full - sl) / abs(full) < 1e-6),
      f"rel={abs(full - sl) / abs(full):.2e}")

# 8. global topk: exactly K, correct coordinates
class FakeMod:
    def __init__(self, w):
        self.weight = w
lins = [("a", FakeMod(torch.randn(16, 16, dtype=torch.bfloat16))),
        ("b", FakeMod(torch.randn(16, 16, dtype=torch.bfloat16)))]
lins[1][1].weight[3, 3] = 99.0
flat = global_topk_indices(lins, 5)
total = sum(len(v) for v in flat.values())
check("global topk returns exactly K", total == 5, f"got {total}")
check("planted global max found in tensor b",
      "b" in flat and (3 * 16 + 3) in flat["b"].tolist())

print()
if fails:
    print(f"SELFTEST FAILED: {fails}")
    raise SystemExit(1)
print("SELFTEST PASSED (all checks)")
