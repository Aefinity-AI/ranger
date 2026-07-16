"""
RANGER Phase-1 — shared utilities for testing the theory on a REAL model.

Everything here is CPU-safe (no CUDA required) so the 135M model runs on a laptop.
Pure-PyTorch Fast Walsh–Hadamard Transform included so no compiled kernel is needed.
"""
import numpy as np
import torch
import torch.nn as nn


def load(model_id, dtype=None, device=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype or (torch.float16 if device == "cuda" else torch.float32)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    return model, tok, device


# ---------------------------------------------------------------- Hadamard / FWHT
def hadamard(x):
    """Normalized Fast Walsh–Hadamard Transform over the last dim (power of 2)."""
    n = x.shape[-1]
    assert (n & (n - 1)) == 0, f"FWHT needs power-of-two length, got {n}"
    shape = x.shape
    x = x.reshape(-1, n).clone()
    h = 1
    while h < n:
        x = x.view(-1, n // (2 * h), 2, h)
        a, b = x[:, :, 0, :], x[:, :, 1, :]
        x = torch.stack((a + b, a - b), dim=2).reshape(-1, n)
        h *= 2
    return (x / np.sqrt(n)).reshape(shape)


def next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def hadamard_pad(x):
    """FWHT with zero-pad to next power of two on the last dim; returns (y, orig_n)."""
    n = x.shape[-1]
    m = next_pow2(n)
    if m != n:
        x = torch.nn.functional.pad(x, (0, m - n))
    return hadamard(x), n


# ---------------------------------------------------------------- quantizers
def quantize_weight(W, bits, per="out"):
    """Symmetric uniform RTN. per='out' scales per output-channel (row of [out,in])."""
    qmax = 2 ** (bits - 1) - 1
    if per == "out":
        s = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    elif per == "in":
        s = W.abs().amax(dim=0, keepdim=True).clamp_min(1e-8) / qmax
    else:
        s = W.abs().max().clamp_min(1e-8) / qmax
    return (W / s).round().clamp(-qmax - 1, qmax) * s


def quantize_act(x, bits, per_token=True):
    qmax = 2 ** (bits - 1) - 1
    dim = -1 if per_token else None
    if per_token:
        s = x.abs().amax(dim=dim, keepdim=True).clamp_min(1e-8) / qmax
    else:
        s = x.abs().max().clamp_min(1e-8) / qmax
    return (x / s).round().clamp(-qmax - 1, qmax) * s


# ---------------------------------------------------------------- metrics
def tensor_stats(x):
    """Outlier signature of a [tokens, dim] activation matrix."""
    x = x.detach().float().reshape(-1, x.shape[-1])
    flat = x.reshape(-1)
    mu, sd = flat.mean(), flat.std().clamp_min(1e-9)
    kurt = (((flat - mu) / sd) ** 4).mean().item() - 3.0
    maxabs = flat.abs().max().item()
    incoh = maxabs * np.sqrt(flat.numel()) / (flat.norm().item() + 1e-9)
    # per-channel max magnitude -> how concentrated are the outliers?
    ch_max = x.abs().amax(dim=0)
    top = torch.topk(ch_max, k=min(5, ch_max.numel())).values.tolist()
    return dict(kurtosis=kurt, max_abs=maxabs, incoherence=incoh,
                rms=flat.pow(2).mean().sqrt().item(), top_channels=top)


# ---------------------------------------------------------------- perplexity
@torch.no_grad()
def wikitext_ppl(model, tok, device, seqlen=1024, max_windows=40, stride=None):
    """Sliding-window WikiText-2 perplexity (HF recipe, context-masked)."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    stride = stride or seqlen
    nlls, prev, n = [], 0, enc.size(0)
    for begin in range(0, n, stride):
        end = min(begin + seqlen, n)
        trg = end - prev
        ids = enc[begin:end].unsqueeze(0).to(device)
        tgt = ids.clone()
        tgt[:, :-trg] = -100
        loss = model(ids, labels=tgt).loss
        nlls.append(loss.float() * trg)
        prev = end
        if end == n or len(nlls) >= max_windows:
            break
    return float(torch.exp(torch.stack(nlls).sum() / prev))


@torch.no_grad()
def collect_hidden_stats(model, tok, device, text_tokens=4096, seqlen=1024):
    """Per-layer residual-stream outlier stats via output_hidden_states."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0][:text_tokens]
    ids = enc[:seqlen].unsqueeze(0).to(device)
    out = model(ids, output_hidden_states=True)
    return [tensor_stats(h[0]) for h in out.hidden_states]  # list over layers (+embed)


def linears(model, skip=("lm_head",)):
    """Yield (name, module) for quantizable nn.Linear layers (skips head/embeddings)."""
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and not any(s in name for s in skip):
            yield name, m
