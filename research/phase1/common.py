"""Shared harness for Phase 1b (E8-E12).

Bit-compatible with E7's methodology: same wikitext-2 slice, same
non-overlapping ctx-1024 perplexity windows. One deliberate change:
log_softmax is computed in 256-row slices, which caps the fp32 logits
transient (~100 MB instead of ~400 MB on SmolLM2, ~1.2 GB on Qwen3-0.6B)
without changing the NLL — log_softmax is row-wise.

Audit findings baked in (workflow wf_4b49c9dc-3d0, 2026-07-16):
- outlier protection must EXCLUDE entries from the scale, not restore values
  (per-channel absmax reconstructs each row max bit-exactly, so value
  restoration is a no-op by construction);
- protected entries are selected by topk INDICES (exactly K), never by a
  >=-threshold comparison (bf16 magnitude ties restore more than K);
- every protection arm reports n_changed = count of protected entries whose
  value differs from what the arm's quantizer would have produced; an arm
  with n_changed == 0 is invalid by construction, not a result.
"""
import json
import math
import os
import tempfile

import numpy as np
import torch
import torch.nn as nn


def load(model_id, dtype=None, device=None):
    """Load a HF causal-LM + tokenizer (network required; not used by selftests)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype or (torch.float16 if device == "cuda" else torch.float32)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    return model, tok, device


# ---------------------------------------------------------------- Hadamard / FWHT
def hadamard(x):
    """Normalized Fast Walsh-Hadamard Transform over the last dim (power of 2)."""
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


# ---------------------------------------------------------------- generic quantizers
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


def tensor_stats(x):
    """Outlier signature of a [tokens, dim] activation matrix."""
    x = x.detach().float().reshape(-1, x.shape[-1])
    flat = x.reshape(-1)
    mu, sd = flat.mean(), flat.std().clamp_min(1e-9)
    kurt = (((flat - mu) / sd) ** 4).mean().item() - 3.0
    maxabs = flat.abs().max().item()
    incoh = maxabs * np.sqrt(flat.numel()) / (flat.norm().item() + 1e-9)
    ch_max = x.abs().amax(dim=0)
    top = torch.topk(ch_max, k=min(5, ch_max.numel())).values.tolist()
    return dict(kurtosis=kurt, max_abs=maxabs, incoherence=incoh,
                rms=flat.pow(2).mean().sqrt().item(), top_channels=top)


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


def get_eval_ids(tokenizer, n_tokens):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(r["text"] for r in ds)
    return tokenizer(text, return_tensors="pt").input_ids[0][:n_tokens]


def perplexity(model, ids, ctx=1024, logit_slice=256):
    """E7's non-overlapping-window teacher-forced ppl, sliced log_softmax."""
    nll, count = 0.0, 0
    for i in range(0, len(ids) - 1, ctx):
        chunk = ids[i:i + ctx + 1]
        if len(chunk) < 2:
            break
        logits = model(chunk[:-1].unsqueeze(0), use_cache=False).logits[0]
        targets = chunk[1:]
        for j in range(0, logits.shape[0], logit_slice):
            lg = torch.log_softmax(logits[j:j + logit_slice].float(), dim=-1)
            t = targets[j:j + logit_slice]
            nll -= lg[torch.arange(len(t)), t].sum().item()
        count += len(chunk) - 1
        del logits  # drop the ~200MB fp32 tensor before the next forward
    return math.exp(nll / count)


def target_linears(model):
    """All transformer-block linear weights (skip embeddings / lm_head)."""
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and "lm_head" not in n]


def rtn_w4_perchannel_(weight):
    """VERBATIM E7 quantizer (continuity anchor — do not modify)."""
    qmax = 7
    w = weight.to(torch.float32)
    scale = w.abs().amax(dim=1, keepdim=True) / qmax
    scale[scale == 0] = 1.0
    weight.copy_((torch.clamp(torch.round(w / scale), -8, 7) * scale).to(weight.dtype))


def rtn_w4_grouped_(weight, g=128, holdout_mask=None, clip_grid=None):
    """Group-wise (along the input dim) symmetric absmax W4 RTN, fp32 math,
    result stored back in the weight's own dtype.

    holdout_mask : bool tensor, same shape. True entries are EXCLUDED from
        the group absmax (the scale tightens) and then restored to their
        original value — the Super Weight paper's hold-out protocol.
    clip_grid : iterable of clip factors c <= 1. Per group, pick the c that
        minimizes the group's quantization MSE over non-held entries
        (the paper's z-clip analogue, MSE-searched).

    Returns stats: n_groups_per_row (bpw accounting), n_held, n_changed
    (held entries the quantizer WOULD have altered — must be > 0 for the
    hold-out to be a real intervention), scale_shrinkage (mean
    amax_incl/amax_excl over groups that contain held entries; > 1 means
    the hold-out actually tightened scales).
    """
    qmax = 7
    w = weight.to(torch.float32)
    orig = w.clone()
    rows, cols = w.shape
    held = (holdout_mask if holdout_mask is not None
            else torch.zeros(rows, cols, dtype=torch.bool))
    n_changed = 0
    shrink_num, shrink_den = 0.0, 0
    bounds = list(range(0, cols, g))
    for start in bounds:
        sl = slice(start, min(start + g, cols))
        wg = w[:, sl]
        hg = held[:, sl]
        amax_incl = wg.abs().amax(dim=1, keepdim=True)
        amax = wg.abs().masked_fill(hg, 0.0).amax(dim=1, keepdim=True)
        amax[amax == 0] = 1.0
        if hg.any():
            rows_hit = hg.any(dim=1)
            shrink_num += (amax_incl[rows_hit] / amax[rows_hit]).sum().item()
            shrink_den += int(rows_hit.sum().item())
        grid = list(clip_grid) if clip_grid else [1.0]
        best_q, best_mse = None, None
        for c in grid:
            scale = c * amax / qmax
            q = torch.clamp(torch.round(wg / scale), -8, 7) * scale
            mse = ((q - wg).masked_fill(hg, 0.0) ** 2).sum(dim=1, keepdim=True)
            if best_mse is None:
                best_q, best_mse = q, mse
            else:
                better = mse < best_mse
                best_q = torch.where(better, q, best_q)
                best_mse = torch.where(better, mse, best_mse)
        q = best_q
        if hg.any():
            n_changed += int((q.to(weight.dtype)[hg]
                              != orig[:, sl].to(weight.dtype)[hg]).sum().item())
        q = q.clone()
        q[hg] = wg[hg]  # restore held originals exactly
        w[:, sl] = q
    weight.copy_(w.to(weight.dtype))
    return {
        "n_groups_per_row": len(bounds),
        "n_held": int(held.sum().item()),
        "n_changed": n_changed,
        "scale_shrinkage": (shrink_num / shrink_den) if shrink_den else None,
    }


def effective_bpw(cols, n_groups_per_row, scale_bits=16):
    return 4.0 + scale_bits * n_groups_per_row / cols


def orthogonal_q(n, seed=0):
    """Seeded random orthogonal (QR, fp64 build, fp32 return). NOT a fast
    Hadamard — deployment-cost claims must carry that qualifier."""
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(n, n, generator=gen, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r))
    return q.to(torch.float32)


def fake_quant_pertoken(x, bits, exempt_idx=None):
    """Dynamic per-token symmetric absmax fake-quant on the last dim
    (the QuaRot/SpinQuant A-bit convention). bits >= 16 = passthrough.

    exempt_idx: 1-D long tensor of channels EXCLUDED from the absmax and
    passed through at full precision (E11 static exemption)."""
    if bits >= 16:
        return x
    qmax = 2 ** (bits - 1) - 1
    xf = x.to(torch.float32)
    if exempt_idx is not None:
        absx = xf.abs()
        absx[..., exempt_idx] = 0.0
        scale = absx.amax(dim=-1, keepdim=True) / qmax
    else:
        scale = xf.abs().amax(dim=-1, keepdim=True) / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(xf / scale), -qmax - 1, qmax) * scale
    if exempt_idx is not None:
        q[..., exempt_idx] = xf[..., exempt_idx]
    return q.to(x.dtype)


def load_or_init(out_path, meta):
    """Resume support: if out_path exists and its metadata matches `meta`,
    return it (completed arms are skipped by the caller); on mismatch the
    old file is backed up, never silently clobbered."""
    import time
    if os.path.exists(out_path):
        try:
            old = json.load(open(out_path))
        except Exception:
            old = None
        if old is not None and all(old.get(k) == v for k, v in meta.items()):
            return old
        if old is not None:
            bak = out_path + time.strftime(".bak-%Y%m%d%H%M%S")
            os.rename(out_path, bak)
            print(f"metadata mismatch: backed up old results to {bak}",
                  flush=True)
    return {**meta, "arms": []}


def checkpoint_json(path, obj):
    """Atomic JSON write so a killed run never leaves a torn file."""
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def global_topk_indices(linears, k):
    """Exactly-K global |w| selection, returned as {tensor_name: flat_idx
    LongTensor}. Selection by INDEX, not threshold (bf16 tie bug fix).
    Memory-safe: per-tensor top-k candidates first (the global top-k is
    always among them), never a full concatenation — a 440M-param model
    would need an ~880 MB cat otherwise."""
    cand_vals, cand_idx, cand_names = [], [], []
    for n, m in linears:
        w = m.weight.abs().ravel()
        kk = min(k, w.numel())
        v, i = torch.topk(w.float(), kk)
        cand_vals.append(v)
        cand_idx.append(i)
        cand_names.extend([n] * kk)
        del w
    vals = torch.cat(cand_vals)
    idx = torch.cat(cand_idx)
    top = torch.topk(vals, k).indices
    per = {}
    for t in top.tolist():
        per.setdefault(cand_names[t], []).append(int(idx[t]))
    return {n: torch.tensor(sorted(v), dtype=torch.long)
            for n, v in per.items()}
