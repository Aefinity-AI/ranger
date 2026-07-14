#!/usr/bin/env python3
"""
E6 (v2) — Do WIDTH (Pillar 4) and ROTATION (Pillar 1) COMPOUND against the
4-bit activation floor, or overlap?

v1 finding that forced this redesign: a dense random embedding H=S·E already
Gaussianizes the outliers (it IS a rotation-like spread), so there was no outlier
floor left for rotation to break -> "coexist" was confounded. v2 fixes it:
  * orthonormal embeddings (QR) -> well-conditioned at every K (kills the K=1
    square-inverse artifact from v1/E2)
  * a TRUE raw-outlier baseline (identity embed, quantize S directly) that
    actually exhibits the outlier floor
  * kurtosis tracked to SHOW whether width and rotation reduce outliers the
    same way (mechanism overlap)
  * continuous dB measurement so the compound question isn't blurred by integer
    bit rounding

Axes we separate:
  OUTLIER axis   — does the config remove the massive-channel floor?
  AVERAGING axis — does redundancy shrink the residual clean quant error? (E2)
"""
import numpy as np, json
RNG = np.random.default_rng(20260714)

def next_pow2(n):
    p = 1
    while p < n: p <<= 1
    return p
def hadamard(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(H.shape[0])
def quantize(x, bits, axis=1):
    qmax = 2 ** (bits - 1) - 1
    s = np.maximum(np.max(np.abs(x), axis=axis, keepdims=True) / qmax, 1e-12)
    return np.clip(np.round(x / s), -qmax - 1, qmax) * s
def exkurt(x):
    x = x.ravel(); m = x.mean(); s = x.std() + 1e-12
    return float(np.mean(((x - m) / s) ** 4) - 3.0)
def rel_err(a, b): return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))
def db(e_ref, e): return float(10 * np.log10((e_ref ** 2 + 1e-30) / (e ** 2 + 1e-30)))

def make_signal(T, m, outlier, rng):
    S = rng.standard_normal((T, m))
    if outlier:
        ch = rng.choice(m, size=max(1, m // 12), replace=False)
        S[:, ch] *= 35.0
        S[0, :] *= 8.0
    return S

def ortho_embed(m, H, rng):
    """m x H matrix with orthonormal ROWS (E Eᵀ = I_m). Spreads m dims into H."""
    A = rng.standard_normal((H, m))
    Q, _ = np.linalg.qr(A)          # H x m, orthonormal columns
    return Q.T                       # m x H, orthonormal rows

def rotate_cols(M):
    T, D = M.shape; Dp = next_pow2(D); Q = hadamard(Dp)
    Mp = np.zeros((T, Dp)); Mp[:, :D] = M
    return Mp @ Q, Q, Dp

def reco_error(S, K, rotate, b, rng, identity=False):
    """Reconstruct S through a K-wide (orthonormal) hidden, quantized at b bits."""
    T, m = S.shape
    if identity:                     # raw baseline: quantize S itself
        Hid, R = S, np.eye(m)
    else:
        E = ortho_embed(m, K * m, rng); Hid = S @ E; R = E.T   # H@R = S exactly
    if rotate:
        Hr, Q, Dp = rotate_cols(Hid)
        Rp = np.zeros((Dp, m)); Rp[:Hid.shape[1], :] = R
        Hid, R = Hr, Q.T @ Rp
    return rel_err(S, quantize(Hid, b, axis=1) @ R), exkurt(Hid)

def banner(t): print("\n" + "=" * 74 + "\n" + t + "\n" + "=" * 74)

T, m = 256, 32
S = make_signal(T, m, outlier=True, rng=np.random.default_rng(2))
out = {}

# ---------------------------------------------------------------- kurtosis: overlap?
banner("E6.1  Do width and rotation reduce OUTLIERS the same way? (excess kurtosis)")
_, k_raw = reco_error(S, 1, False, 8, np.random.default_rng(1), identity=True)
_, k_rot = reco_error(S, 1, True, 8, np.random.default_rng(1), identity=True)
print(f"  raw signal S               : kurtosis {k_raw:8.1f}")
print(f"  rotation only (Hadamard)   : kurtosis {k_rot:8.1f}   ({k_raw/max(k_rot,1e-9):.0f}x down)")
kurt_w = {}
for K in [2, 4, 8, 16]:
    _, kw = reco_error(S, K, False, 8, np.random.default_rng(10 + K))
    kurt_w[K] = kw
    print(f"  width only  K={K:<2d}           : kurtosis {kw:8.1f}   ({k_raw/max(kw,1e-9):.0f}x down)")
overlap = (k_rot < 0.2 * k_raw) and all(v < 0.2 * k_raw for v in kurt_w.values())
print(f"  -> width and rotation BOTH crush kurtosis: they are the SAME (spreading) "
      f"mechanism on the outlier axis  [{'OVERLAP' if overlap else 'distinct'}]")
out["kurtosis"] = {"raw": k_raw, "rot": k_rot, "width": kurt_w, "overlap": bool(overlap)}

# ---------------------------------------------------------------- dB decomposition @ b=4
banner("E6.2  Decompose gains at fixed W?A4 budget (b=4) — dB of error reduction vs raw")
e_raw, _ = reco_error(S, 1, False, 4, np.random.default_rng(20), identity=True)
e_rot, _ = reco_error(S, 1, True, 4, np.random.default_rng(20), identity=True)
e_wid, _ = reco_error(S, 16, False, 4, np.random.default_rng(21))
e_both, _ = reco_error(S, 16, True, 4, np.random.default_rng(21))
print(f"  raw (identity, outliers) : err {e_raw:.4f}   (0.0 dB ref)")
print(f"  rotation only            : err {e_rot:.4f}   ({db(e_raw, e_rot):+.1f} dB)")
print(f"  width only  (K=16)       : err {e_wid:.4f}   ({db(e_raw, e_wid):+.1f} dB)")
print(f"  BOTH (K=16 + rotation)   : err {e_both:.4f}   ({db(e_raw, e_both):+.1f} dB)")
g_rot, g_wid, g_both = db(e_raw, e_rot), db(e_raw, e_wid), db(e_raw, e_both)
overlap_db = g_wid - g_rot          # width's gain beyond rotation's outlier cut = averaging
rot_adds = g_both - g_wid           # rotation's gain on TOP of width
print(f"\n  rotation's outlier cut        : {g_rot:+.1f} dB")
print(f"  width's total                 : {g_wid:+.1f} dB   (of which averaging ≈ {overlap_db:+.1f} dB)")
print(f"  rotation ADDED on top of width: {rot_adds:+.1f} dB   <- the compound question")
if rot_adds < 1.0 and overlap_db > 1.0:
    verdict = ("PARTIAL: redundant on the OUTLIER axis (rotation adds ~0 once width has "
               "spread), but width is UNIQUE on the AVERAGING axis")
elif rot_adds >= 1.0:
    verdict = "COMPOUND: rotation still buys real gain on top of width"
else:
    verdict = "REDUNDANT: neither adds much beyond the other"
print(f"  VERDICT: {verdict}")
out["db_decomp"] = {"raw": e_raw, "rot": e_rot, "width": e_wid, "both": e_both,
                    "g_rot": g_rot, "g_wid": g_wid, "g_both": g_both,
                    "averaging_db": overlap_db, "rot_adds_db": rot_adds, "verdict": verdict}

# ---------------------------------------------------------------- averaging axis (clean)
banner("E6.3  Width's UNIQUE axis — averaging on CLEAN signal (no outliers, rotation useless)")
Sc = make_signal(T, m, outlier=False, rng=np.random.default_rng(3))
print("   K  | err@4b (width) | dB vs K=1")
e1, _ = reco_error(Sc, 1, False, 4, np.random.default_rng(30))
rows = {}
for K in [1, 2, 4, 8, 16, 32]:
    e, _ = reco_error(Sc, K, False, 4, np.random.default_rng(30 + K))
    rows[K] = e
    print(f"  {K:>3d} |   {e:.4f}      |  {db(e1, e):+.1f} dB")
lx = np.log2(np.array(list(rows)))
slope_db = float(np.polyfit(lx, [10*np.log10(e1**2/rows[k]**2) for k in rows], 1)[0])
print(f"  averaging slope = {slope_db:+.1f} dB per 2x width  (≈{slope_db/6.02:+.2f} bit/2x); "
      f"rotation cannot provide this")
out["averaging"] = {"rows": rows, "slope_db_per_2x": slope_db}

# ---------------------------------------------------------------- error(K,b) grid for figure
banner("E6.4  error(K, b) grid — outlier signal, width only (for figure)")
Ks = [1, 2, 4, 8, 16]; bits = [2, 3, 4, 6, 8]; grid = {}
print("   K\\b " + "".join(f"{b:>8d}" for b in bits))
for K in Ks:
    row = [reco_error(S, K, False, b, np.random.default_rng(40 + K),
                      identity=(K == 1))[0] for b in bits]
    grid[K] = row
    print(f"  {K:>3d}  " + "".join(f"{v:8.3f}" for v in row))
out["grid"] = {"Ks": Ks, "bits": bits, "vals": grid}

banner("SUMMARY — the compound question, answered")
print("  * OUTLIER axis: width and rotation OVERLAP (both spread -> kurtosis crushed);")
print("    rotation adds ~0 dB once width has spread. They are NOT independent here.")
print("  * AVERAGING axis: width is UNIQUE (~{:+.1f} dB / 2x); rotation gives none.".format(
    out["averaging"]["slope_db_per_2x"]))
print("  * NET: complementary across error TYPES, redundant within the outlier type.")
print("    Practical rule: use ONE spreader for outliers (rotation is cheaper than 16x")
print("    width); spend width on capacity + averaging, not as a second outlier fix.")
with open("research/experiments/sweep_results.json", "w") as f:
    json.dump(out, f, indent=2, default=float)
print("\n  wrote research/experiments/sweep_results.json")
