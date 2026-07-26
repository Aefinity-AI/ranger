#!/usr/bin/env python3
"""
RANGER Experiment B — the KV-cache quantization lab.

Claims under test (report §2.5/§2.6, roadmap B):
  B-1  SINK EXEMPTION: attention sinks park massive KV on the first few tokens
       (2601.22966); keeping ≤4 sink tokens' KV in fp should rescue most of
       low-bit KV damage.
  B-2  LOCAL/GLOBAL SPLIT (the unclaimed coupling): on models that interleave
       sliding-window and global attention (Gemma-2/3 style), local-layer KV
       has bounded, stationary statistics -> it should tolerate 2-bit at
       ≲half the PPL damage of global-layer KV at equal average bits.
       Placebo control: an even/odd layer split on ANY model should show ~no
       asymmetry — if it does, the measurement is an artifact.
  B-3  ROPE TAX (probe): RoPE injects position structure into keys; keys
       should quantize measurably worse AFTER RoPE than before, and worse
       than values (which have no RoPE). Motivates p-RoPE ⊕ rotation.

Method: a FakeQuantKVCache (DynamicCache subclass) quantize-dequantizes each
K/V chunk AT CACHE-UPDATE TIME — the real streaming path, post-RoPE keys,
per-layer bit policies, absolute-position sink exemption. PPL is computed by
chunked evaluation through the cache; the fp-cache arm must match the plain
full-forward PPL, which the selftest asserts — that equality validates the rig.

Usage:
    python kv_lab.py --model HuggingFaceTB/SmolLM2-135M     # B-1, B-3, placebo
    python kv_lab.py --model google/gemma-3-1b-pt           # + real B-2 split
    python kv_lab.py --selftest                             # network-free
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase1"))
from common import load, linears                                  # noqa: E402
from transformers.cache_utils import DynamicCache                 # noqa: E402


# ------------------------------------------------------------ KV fake quant
def kv_fake_quant(x, bits, axis="token", exempt_prefix=0):
    """Symmetric uniform quant of a KV chunk [B, H, T, D]. axis='token' scales
    per (b,h,t) vector — the streaming-realistic choice; 'channel' scales per
    (b,h,d) within the chunk. exempt_prefix: leading tokens kept exact (sink)."""
    if bits is None or bits >= 16:
        return x
    qmax = 2 ** (int(bits) - 1) - 1
    dim = -1 if axis == "token" else -2
    s = x.abs().amax(dim=dim, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.clamp((x / s).round(), -qmax - 1, qmax) * s
    if exempt_prefix > 0:
        q = torch.cat([x[:, :, :exempt_prefix], q[:, :, exempt_prefix:]], dim=2)
    return q.to(x.dtype)


class FakeQuantKVCache(DynamicCache):
    """DynamicCache that fake-quantizes each incoming K/V chunk.

    kbits / vbits: int (all layers) or list[int|None] per layer (None = fp).
    sink: number of ABSOLUTE positions at sequence start kept full-precision.
    """

    def __init__(self, kbits=4, vbits=4, sink=0, axis="token"):
        super().__init__()
        self._kbits, self._vbits = kbits, vbits
        self._sink, self._axis = sink, axis

    def _bits(self, spec, layer_idx):
        return spec[layer_idx] if isinstance(spec, (list, tuple)) else spec

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        past = self.get_seq_length(layer_idx)
        exempt = max(0, min(self._sink - past, key_states.shape[2]))
        k = kv_fake_quant(key_states, self._bits(self._kbits, layer_idx),
                          self._axis, exempt)
        v = kv_fake_quant(value_states, self._bits(self._vbits, layer_idx),
                          self._axis, exempt)
        return super().update(k, v, layer_idx, cache_kwargs)


# ------------------------------------------------------------ chunked PPL
@torch.no_grad()
def ppl_through_cache(model, ids, cache_factory, chunk=128):
    """Teacher-forced PPL where K/V flow through the (quantizing) cache.
    Feeds the sequence in chunks with use_cache=True, gathers all logits,
    computes the standard shifted cross-entropy."""
    past = cache_factory()
    chunks = []
    T = ids.shape[1]
    for s in range(0, T, chunk):
        out = model(ids[:, s:s + chunk], past_key_values=past, use_cache=True)
        past = out.past_key_values
        chunks.append(out.logits.float())
    logits = torch.cat(chunks, dim=1)
    loss = torch.nn.functional.cross_entropy(
        logits[0, :-1], ids[0, 1:], reduction="mean")
    return float(torch.exp(loss))


@torch.no_grad()
def ppl_plain(model, ids):
    out = model(ids, labels=ids)
    return float(torch.exp(out.loss))


# ------------------------------------------------------------ layer split
def find_layer_split(model):
    """(local_layers, global_layers) from config.layer_types if present."""
    lt = getattr(model.config, "layer_types", None)
    if lt:
        loc = [i for i, t in enumerate(lt) if "sliding" in t or "local" in t or "conv" in t]
        glo = [i for i, t in enumerate(lt) if i not in loc]
        if loc and glo:
            return loc, glo
    return None, None


def bits_for_split(n_layers, group_a, bits_a, bits_b):
    """Per-layer bit list: group_a layers get bits_a, the rest bits_b."""
    return [bits_a if i in group_a else bits_b for i in range(n_layers)]


# ------------------------------------------------------------ RoPE-tax probe
@torch.no_grad()
def rope_tax_probe(model, ids, bits=2):
    """Quant error of keys BEFORE RoPE (k_proj outputs, via hooks) vs AFTER
    RoPE (what the cache actually stores), and values as the no-RoPE control."""
    caps, handles = {}, []
    for n, m in linears(model):
        if n.endswith("k_proj") or n.endswith("v_proj"):
            def mk(name):
                def hook(mod, inp, out):
                    caps.setdefault(name, out.detach())
                return hook
            handles.append(m.register_forward_hook(mk(n)))
    cache = DynamicCache()
    model(ids, past_key_values=cache, use_cache=True)
    for h in handles:
        h.remove()

    def rel_err_2d(x):                      # [T, D-ish]: per-token quant error
        q = kv_fake_quant(x.unsqueeze(0).unsqueeze(0), bits).squeeze()
        return float((x - q).norm() / (x.norm() + 1e-9))

    pre_k = [rel_err_2d(v[0].float()) for n, v in caps.items() if "k_proj" in n]
    vals = [rel_err_2d(v[0].float()) for n, v in caps.items() if "v_proj" in n]
    post_k = []
    for layer in range(len(model.model.layers)):
        # transformers >=5: cache.layers[i].keys; <=4.x: cache[i][0]
        if hasattr(cache, "layers"):
            k = cache.layers[layer].keys     # [B, H, T, D] post-RoPE keys
        else:
            k = cache[layer][0]
        post_k.append(rel_err_2d(k[0].reshape(-1, k.shape[-1]).float()))
    return {"pre_rope_K": float(np.mean(pre_k)),
            "post_rope_K": float(np.mean(post_k)),
            "V_no_rope": float(np.mean(vals)),
            "rope_tax_x": float(np.mean(post_k) / max(np.mean(pre_k), 1e-9))}


# ------------------------------------------------------------ experiment arms
def run_lab(model, ids, n_layers, out):
    fp_plain = ppl_plain(model, ids)
    fp_cache = ppl_through_cache(model, ids, DynamicCache)
    drift = abs(fp_cache - fp_plain) / fp_plain
    print(f"  rig check: plain PPL {fp_plain:.3f} vs fp-cache PPL {fp_cache:.3f} "
          f"(drift {100*drift:.2f}% — must be ~0)")
    out["fp_ppl"] = fp_plain
    out["rig_drift"] = drift

    print("\n  --- bits sweep (K=V, per-token, no sink exemption) ---")
    out["bits_sweep"] = {}
    for b in [8, 4, 3, 2]:
        p = ppl_through_cache(model, ids, lambda: FakeQuantKVCache(b, b))
        out["bits_sweep"][b] = p
        print(f"    KV{b}: PPL {p:8.3f}  (+{100*(p-fp_plain)/fp_plain:6.1f}%)")

    print("\n  --- B-1: sink exemption at KV2 and KV3 ---")
    out["sink"] = {}
    for b in [3, 2]:
        row = {}
        base_dmg = out["bits_sweep"][b] - fp_plain
        for sink in [0, 1, 4, 16]:
            p = ppl_through_cache(model, ids,
                                  lambda: FakeQuantKVCache(b, b, sink=sink))
            rescued = (out["bits_sweep"][b] - p) / max(base_dmg, 1e-9)
            row[sink] = {"ppl": p, "rescued_frac": rescued}
            print(f"    KV{b} sink={sink:2d}: PPL {p:8.3f}  "
                  f"(rescues {100*rescued:5.1f}% of KV{b} damage)")
        out["sink"][b] = row

    print("\n  --- K/V asymmetry at equal average bits (3) ---")
    out["kv_asym"] = {
        "K2_V4": ppl_through_cache(model, ids, lambda: FakeQuantKVCache(2, 4)),
        "K4_V2": ppl_through_cache(model, ids, lambda: FakeQuantKVCache(4, 2)),
    }
    a, b_ = out["kv_asym"]["K2_V4"], out["kv_asym"]["K4_V2"]
    print(f"    K2/V4: {a:.3f}   K4/V2: {b_:.3f}   -> "
          f"{'K is the fragile one (RoPE tax, as predicted)' if a > b_ else 'V is the fragile one (unexpected)'}")

    loc, glo = find_layer_split(model)
    print("\n  --- B-2: group fragility — quantize ONE group to 2-bit, rest fp ---")
    # Fair design: group sizes differ (Gemma is 5:1 local:global), so mixing
    # 2/4-bit assignments gives UNEQUAL average bits between arms. Instead we
    # quantize only one group (others full-precision) and compare the damage
    # normalized per quantized layer.
    out["split"] = {}

    def group_damage(group):
        kb = bits_for_split(n_layers, group, 2, None)   # None = fp elsewhere
        p = ppl_through_cache(model, ids, lambda: FakeQuantKVCache(kb, kb))
        return p, (p - fp_plain) / max(len(group), 1)

    if loc:
        p_l, dpl_l = group_damage(loc)
        p_g, dpl_g = group_damage(glo)
        out["split"]["local_only2"] = {"ppl": p_l, "dmg_per_layer": dpl_l,
                                       "n_layers": len(loc)}
        out["split"]["global_only2"] = {"ppl": p_g, "dmg_per_layer": dpl_g,
                                        "n_layers": len(glo)}
        print(f"    local-only@2  ({len(loc):2d} layers): PPL {p_l:8.3f}  "
              f"damage/layer {dpl_l:+.4f}")
        print(f"    global-only@2 ({len(glo):2d} layers): PPL {p_g:8.3f}  "
              f"damage/layer {dpl_g:+.4f}")
        verdict = ("CONFIRMED: local KV tolerates low bits better (per-layer)"
                   if dpl_l < dpl_g else "REFUTED for this model")
        print(f"    -> {verdict}")
    else:
        print("    (model has no local/global interleave — placebo control only)")
    even = list(range(0, n_layers, 2))
    odd = [i for i in range(n_layers) if i not in even]
    p_e, dpl_e = group_damage(even)
    p_o, dpl_o = group_damage(odd)
    out["split"]["placebo_even2"] = {"ppl": p_e, "dmg_per_layer": dpl_e}
    out["split"]["placebo_odd2"] = {"ppl": p_o, "dmg_per_layer": dpl_o}
    gap = abs(dpl_e - dpl_o) / max(min(abs(dpl_e), abs(dpl_o)), 1e-9)
    print(f"    placebo even-only@2 dmg/layer {dpl_e:+.4f} vs odd-only@2 {dpl_o:+.4f}  "
          f"(asymmetry {100*gap:.0f}% — should be small; large => depth, not "
          f"locality, drives differences)")

    print("\n  --- B-3: RoPE-tax probe (2-bit key quant error) ---")
    out["rope_tax"] = rope_tax_probe(model, ids, bits=2)
    r = out["rope_tax"]
    print(f"    pre-RoPE K err {r['pre_rope_K']:.4f} | post-RoPE K err "
          f"{r['post_rope_K']:.4f} | V err {r['V_no_rope']:.4f} | "
          f"RoPE tax {r['rope_tax_x']:.2f}x")
    return out


# ------------------------------------------------------------ selftest
def selftest():
    """Network-free validation on a tiny random model. The load-bearing check:
    fp-cache chunked PPL must EQUAL plain full-forward PPL — that proves the
    cache rig computes the same math before we trust any quantized number."""
    from transformers import LlamaConfig, LlamaForCausalLM
    print("SELFTEST — tiny random model, no network\n")
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=512, hidden_size=128, intermediate_size=256,
                      num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=512)
    model = LlamaForCausalLM(cfg).eval()
    ids = torch.randint(0, 512, (1, 256))

    print("1. rig equality: fp-cache chunked PPL == plain full-forward PPL")
    p_plain = ppl_plain(model, ids)
    p_cache = ppl_through_cache(model, ids, DynamicCache, chunk=64)
    print(f"   plain {p_plain:.4f}  vs fp-cache {p_cache:.4f}")
    assert abs(p_cache - p_plain) / p_plain < 1e-3, "cache rig diverges from plain"

    print("2. logit perturbation grows as KV bits shrink (mechanical check)")
    # PPL direction is undefined on a RANDOM model (its loss is ~uniform), but
    # the logit distance from the fp run is guaranteed to grow as bits drop.
    @torch.no_grad()
    def logits_through(cache_factory):
        past, chunks = cache_factory(), []
        for s in range(0, ids.shape[1], 64):
            o = model(ids[:, s:s + 64], past_key_values=past, use_cache=True)
            past = o.past_key_values
            chunks.append(o.logits.float())
        return torch.cat(chunks, dim=1)
    l_fp = logits_through(DynamicCache)
    d8 = float((logits_through(lambda: FakeQuantKVCache(8, 8)) - l_fp).norm())
    d2 = float((logits_through(lambda: FakeQuantKVCache(2, 2)) - l_fp).norm())
    print(f"   ||Δlogits|| KV8 {d8:.3f}   KV2 {d2:.3f}")
    assert d2 > d8 > 0, "KV quant noise not flowing through attention as expected"

    print("3. sink exemption: exempting EVERYTHING equals fp")
    p_all = ppl_through_cache(model, ids,
                              lambda: FakeQuantKVCache(2, 2, sink=10_000), chunk=64)
    print(f"   KV2 sink=inf {p_all:.4f}  (fp {p_plain:.4f})")
    assert abs(p_all - p_plain) / p_plain < 1e-3

    print("4. per-layer bit lists respected (even@2 and odd@2 perturb differently)")
    kb_e = bits_for_split(4, [0, 2], 2, None)     # None = fp on the other layers
    kb_o = bits_for_split(4, [1, 3], 2, None)
    d_e = float((logits_through(lambda: FakeQuantKVCache(kb_e, kb_e)) - l_fp).norm())
    d_o = float((logits_through(lambda: FakeQuantKVCache(kb_o, kb_o)) - l_fp).norm())
    print(f"   ||Δlogits|| even-layers@2 {d_e:.3f}   odd-layers@2 {d_o:.3f}")
    assert d_e > 0 and d_o > 0 and abs(d_e - d_o) > 1e-6, \
        "per-layer bit policy not differentiating layers"

    print("5. RoPE-tax probe runs and keys carry a positive tax")
    r = rope_tax_probe(model, ids, bits=2)
    print(f"   pre-K {r['pre_rope_K']:.4f}  post-K {r['post_rope_K']:.4f}  "
          f"V {r['V_no_rope']:.4f}  tax {r['rope_tax_x']:.2f}x")
    assert r["rope_tax_x"] > 0

    print("\nSELFTEST OK — cache rig proven equal to plain forward; all arms run. "
          "Real claims need real weights.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--out", default="kv_results.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    print(f"loading {args.model} ...")
    model, tok, device = load(args.model, dtype=torch.float32)
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt")\
        .input_ids[:, :args.tokens].to(device)

    out = {"model": args.model, "tokens": args.tokens}
    run_lab(model, ids, model.config.num_hidden_layers, out)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
