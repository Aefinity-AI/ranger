#!/usr/bin/env python3
"""
Network-free self-test — validates every Phase-1 code path on a tiny randomly
initialized model, so you can trust the harness before downloading real weights.
(The sandbox that authored this blocks huggingface.co; on your PC the real
scripts download and run against actual pretrained models.)

    python selftest.py
"""
import copy
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from common import (tensor_stats, quantize_weight, quantize_act, hadamard,
                    hadamard_pad, linears)


def main():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=512, hidden_size=128, intermediate_size=256,
                      num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=256)
    model = LlamaForCausalLM(cfg).eval()
    ids = torch.randint(0, 512, (1, 128))

    print("1. forward + hidden-state stats + loss path")
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, labels=ids)
    stats = [tensor_stats(h[0]) for h in out.hidden_states]
    print(f"   hidden layers captured: {len(stats)}   loss: {float(out.loss):.3f}")
    assert len(stats) == cfg.num_hidden_layers + 1

    print("2. FWHT is involutive (H(H(x)) == x)")
    x = torch.randn(4, 128)
    assert torch.allclose(hadamard(hadamard(x)), x, atol=1e-4)
    print("   OK")

    print("3. weight quantizer error shrinks as bits increase")
    W = torch.randn(64, 128)
    prev = 1e9
    for b in [2, 3, 4, 8]:              # ascending bits -> error must decrease
        e = float((W - quantize_weight(W, b)).norm() / W.norm())
        print(f"   W{b}: rel-err {e:.4f}")
        assert e < prev, "more bits must mean less error"; prev = e

    print("4. rotation cuts quant error on OUTLIERED synthetic activations (E1 on tensors)")
    x = torch.randn(256, 128); x[:, [3, 50, 90]] *= 40.0     # massive channels
    Wl = torch.randn(128, 128) / np.sqrt(128)
    Y = x @ Wl.t()
    e_naive = float((Y - quantize_act(x, 4) @ quantize_weight(Wl, 4).t()).norm() / Y.norm())
    xr, _ = hadamard_pad(x); Wr, _ = hadamard_pad(Wl)
    e_rot = float((Y - quantize_act(xr, 4) @ quantize_weight(Wr, 4).t()).norm() / Y.norm())
    print(f"   naive {e_naive:.4f}   rotated {e_rot:.4f}   cut {e_naive/e_rot:.2f}x")
    assert e_rot < e_naive, "rotation should help on outliered activations"

    print("5. importance-ordered vs uniform weight quant (Pillar 2, tiny)")
    n_lin = sum(1 for _ in linears(model))
    W = torch.randn(128, 128); W[torch.randperm(128)[:8]] *= 8   # a few big-norm channels
    b = 3
    e_uni = float((W - quantize_weight(W, b)).norm() / W.norm())
    imp = W.abs().sum(1); hi = torch.argsort(imp, descending=True)[:64]
    Wq = quantize_weight(W, b - 1); Wq[hi] = quantize_weight(W[hi], b + 1)
    e_ord = float((W - Wq).norm() / W.norm())
    print(f"   linears found: {n_lin}   uniform {e_uni:.4f}   ordered {e_ord:.4f}   "
          f"{'ordered wins' if e_ord < e_uni else 'no gain'}")

    print("\nSELFTEST OK — every Phase-1 code path runs. On your PC, the same code")
    print("downloads real weights and produces the real outlier map + PPL numbers.")


if __name__ == "__main__":
    main()
