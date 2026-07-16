#!/usr/bin/env python3
"""
Phase-2 self-test — network-free validation of the QAT scaffold on a tiny
random model. Verifies each mechanism BEHAVES (not just runs):

  1. conversion + forward + effective avg bits accounting
  2. STE: gradients flow through fake-quant to the fp32 master weights
  3. QAT learns THROUGH the quantizer (loss drops on a memorization task)
  4. ternary weights really take only 3 values per channel {-s, 0, +s}
  5. super-weight split: sparse entries exact-fp AND excluded from the dense
     scale (constraint G) — quant error of the dense part must be LOWER than
     quantizing the un-split matrix
  6. nested precision: hi channels carry lower error than lo channels
  7. kurtosis_loss is differentiable and minimizing it reduces kurtosis
  8. rotate path is mathematically exact in fp (xH·(WH)^T == x·W^T)

    python selftest_phase2.py
"""
import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase1"))
from qat_modules import (FakeQuantLinear, convert_to_qat, kurtosis_loss,
                         average_bits, _fake_quant, _superweight_mask)
from train_qat import make_tiny_model


def main():
    torch.manual_seed(0)

    print("1. conversion + forward + avg-bits accounting")
    model, _, _ = make_tiny_model()
    n = convert_to_qat(model, bits=3, superweight_pct=0.002, nested=True,
                       protect_downproj=True)
    ids = torch.randint(0, 512, (2, 64))
    out = model(ids, labels=ids)
    ab = average_bits(model)
    print(f"   converted {n} linears | avg bits {ab:.2f} | loss {float(out.loss):.3f}")
    assert n > 0 and 1.5 < ab < 6.0

    print("2. STE gradient flow to fp32 master weights")
    out.loss.backward()
    fq = next(m for m in model.modules() if isinstance(m, FakeQuantLinear))
    g = fq.weight.grad
    assert g is not None and float(g.abs().sum()) > 0, "no gradient through STE"
    print(f"   grad norm on a fake-quant weight: {float(g.norm()):.4f}  OK")

    print("3. QAT learns THROUGH the quantizer (memorization, 60 steps, W2)")
    model, _, _ = make_tiny_model()
    convert_to_qat(model, bits=2)
    batch = torch.randint(0, 512, (4, 64))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses = []
    for _ in range(60):
        loss = model(batch, labels=batch).loss
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        losses.append(float(loss))
    print(f"   W2 loss: {losses[0]:.3f} -> {losses[-1]:.3f}")
    assert losses[-1] < 0.7 * losses[0], "QAT failed to learn through STE"

    print("4. ternary weights take only values {-s, 0, +s}")
    lin = nn.Linear(64, 32, bias=False)
    fq = FakeQuantLinear(lin, ternary=True)
    W = fq.effective_weight()
    for r in range(0, 32, 8):
        u = torch.unique(W[r])
        assert u.numel() <= 3, f"row {r} has {u.numel()} distinct values"
    print(f"   OK (row 0 uniques: {torch.unique(W[0]).tolist()})")

    print("5. super-weight split (constraint G): exact fp + better dense scale")
    W = torch.randn(64, 64)
    W[torch.randint(0, 64, (4,)), torch.randint(0, 64, (4,))] *= 40.0  # super-weights
    lin = nn.Linear(64, 64, bias=False); lin.weight.data = W.clone()
    mask = _superweight_mask(W, 0.002)
    fq = FakeQuantLinear(lin, bits=3, sw_mask=mask)
    Weff = fq.effective_weight()
    assert torch.equal(Weff[mask], W[mask]), "sparse path must be exact fp"
    e_split = float((W - Weff).norm() / W.norm())
    e_nosplit = float((W - _fake_quant(W, 3)).norm() / W.norm())
    print(f"   err with split {e_split:.4f}  vs un-split {e_nosplit:.4f}")
    assert e_split < e_nosplit, "splitting super-weights must reduce quant error"

    print("6. nested precision: hi channels get lower error than lo channels")
    W = torch.randn(64, 64)
    imp = W.abs().sum(1)
    hi = torch.zeros(64, dtype=torch.bool); hi[torch.argsort(imp, descending=True)[:32]] = True
    lin = nn.Linear(64, 64, bias=False); lin.weight.data = W.clone()
    fq = FakeQuantLinear(lin, bits=3, hi_mask=hi)
    Weff = fq.effective_weight()
    err = (W - Weff).norm(dim=1) / W.norm(dim=1).clamp_min(1e-9)
    e_hi, e_lo = float(err[hi].mean()), float(err[~hi].mean())
    print(f"   per-channel rel-err  hi(4b): {e_hi:.4f}   lo(2b): {e_lo:.4f}")
    assert e_hi < e_lo

    print("7. kurtosis_loss is differentiable and reduces kurtosis when minimized")
    x = torch.randn(1, 64, 128)
    x[..., :3] *= 25.0                       # massive channels
    x = nn.Parameter(x)
    def kurt(t):
        f = t.detach().float().reshape(-1)
        return float((((f - f.mean()) / f.std().clamp_min(1e-9)) ** 4).mean() - 3)
    k0 = kurt(x)
    opt = torch.optim.Adam([x], lr=0.05)
    for _ in range(200):
        loss = kurtosis_loss((torch.zeros_like(x), x))   # [0]=embed is skipped
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    k1 = kurt(x)
    print(f"   kurtosis {k0:.1f} -> {k1:.1f}")
    assert k1 < 0.5 * k0

    print("8. rotate path is exact in fp")
    lin = nn.Linear(96, 48, bias=False)      # 96 -> pads to 128
    x = torch.randn(5, 96)
    y_ref = F.linear(x, lin.weight)
    fq = FakeQuantLinear(lin, bits=16, rotate=True)      # 16b ~ no quant noise
    fq.bits = 16
    y_rot = fq(x)
    err = float((y_ref - y_rot).norm() / y_ref.norm())
    print(f"   rel diff vs unrotated: {err:.2e}")
    assert err < 1e-4

    # --- regression tests for the three review findings -----------------
    print("9. [regression] ternary gradients reach weights OUTSIDE clamp range")
    lin = nn.Linear(64, 32, bias=False)
    lin.weight.data[0, :8] = 5.0                         # |W| >> s = mean|W|
    fq = FakeQuantLinear(lin, ternary=True)
    fq(torch.randn(4, 64)).sum().backward()
    g_out = fq.weight.grad[0, :8]
    assert float(g_out.abs().sum()) > 0, "clamped-out ternary weights got no gradient"
    print(f"   grad on |W|>>s entries: {float(g_out.abs().mean()):.4f}  OK")

    print("10. [regression] binary (1-bit) + super-weight split stays exact")
    W = torch.randn(64, 64); W[2, 3] = 50.0; W[10, 20] = -45.0
    lin = nn.Linear(64, 64, bias=False); lin.weight.data = W.clone()
    mask = _superweight_mask(W, 0.001)
    fq = FakeQuantLinear(lin, bits=1, sw_mask=mask)
    Weff = fq.effective_weight()
    assert torch.equal(Weff[mask], W[mask]), "binary path double-counted super-weights"
    print(f"   sparse entries exact under 1-bit dense quant  OK")

    print("11. [regression] rotation quantizes in the ROTATED basis (and helps)")
    torch.manual_seed(3)
    W = torch.randn(96, 128) / 11.3
    W[:, [5, 40, 77]] *= 30.0                            # outlier in-columns
    x = torch.randn(64, 128)
    lin = nn.Linear(128, 96, bias=False); lin.weight.data = W.clone()
    y_ref = F.linear(x, W)
    e_plain = float((y_ref - FakeQuantLinear(lin, bits=4)(x)).norm() / y_ref.norm())
    e_rot = float((y_ref - FakeQuantLinear(lin, bits=4, rotate=True)(x)).norm() / y_ref.norm())
    print(f"   W4 err  plain {e_plain:.4f}  rotated {e_rot:.4f}  "
          f"({e_plain/e_rot:.2f}x better)")
    assert e_rot < e_plain, "rotated-basis quantization must beat original basis here"

    print("\nSELFTEST OK — every Phase-2 mechanism behaves. On a GPU, run:")
    print("  python train_qat.py --model HuggingFaceTB/SmolLM2-135M --bits 2 \\")
    print("      --superweight-pct 0.005 --nested --protect-downproj --kurt-lambda 0.05")


if __name__ == "__main__":
    main()
