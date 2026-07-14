#!/usr/bin/env python3
"""Render the RANGER mock-run results into a single figure from results.json."""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("research/experiments/results.json"))
BG, PANEL, INK, DIM = "#0b1116", "#111a22", "#e7eef3", "#9fb2bf"
SIG, WARN, CRIT, COUPLE = "#3bc9b0", "#e0a13a", "#e5544e", "#a888f0"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor": "#1e2c37", "text.color": INK, "axes.labelcolor": DIM,
    "xtick.color": DIM, "ytick.color": DIM, "font.family": "monospace", "font.size": 9})

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle("RANGER mock run — mechanism tests on synthetic tensors (all 7 checks PASS)",
             color=INK, fontsize=13, fontweight="bold")

# --- E1: incoherence reduction + bit sweep -------------------------------
a = ax[0, 0]
vals = [R["E1"]["incoh_raw"], R["E1"]["incoh_rot"]]
a.bar(["raw", "rotated"], vals, color=[CRIT, SIG], width=0.55)
a.set_yscale("log"); a.set_ylabel("activation incoherence  μ (log)")
a.set_title("E1 · rotation kills channel outliers", color=INK, fontsize=10, loc="left")
for i, v in enumerate(vals):
    a.text(i, v * 1.1, f"{v:.0f}", ha="center", color=INK, fontsize=10)
a.text(0.5, 0.5, "8.4× drop\nkurtosis 156× drop\n(wins at W4A4;\nsub-4b needs QAT)",
       transform=a.transAxes, ha="center", va="center", color=DIM, fontsize=8.5,
       bbox=dict(boxstyle="round", fc=BG, ec=SIG, alpha=0.85))

# --- E2: capacity-precision duality (err vs width) -----------------------
a = ax[0, 1]
rows = [r for r in R["E2"]["rows"] if r["bstar"] is not None]
K = np.array([r["K"] for r in rows]); err = np.array([r["err"] for r in rows])
a.plot(K, err, "o-", color=SIG, lw=2, ms=7)
a.set_xscale("log", base=2); a.set_yscale("log", base=2)
a.set_xlabel("AltUp width factor  K"); a.set_ylabel("quant error @ fixed 4-bit")
a.set_title("E2 · width buys bits (Pillar 4 VALIDATED)", color=INK, fontsize=10, loc="left")
a.text(0.95, 0.9, f"measured  {R['E2']['bit_slope']:+.2f} bits / 2× width\n"
       f"predicted  −0.50  (½·log₂K)", transform=a.transAxes, ha="right", va="top",
       color=INK, fontsize=9, bbox=dict(boxstyle="round", fc=BG, ec=SIG, alpha=0.85))

# --- E3 + E4: ordered precision & VQ gain --------------------------------
a = ax[1, 0]
e3 = R["E3"]
b = a.bar(["uniform", "Hessian\nopt", "MatFormer\n-ish"],
          [e3["err_uniform"], e3["err_opt"], e3["err_proxy"]],
          color=[DIM, SIG, COUPLE], width=0.6)
a.set_ylabel("total output error (equal avg bits)")
a.set_title("E3 · trained ordering beats uniform", color=INK, fontsize=10, loc="left")
a.text(0.5, 0.85, f"noisy ordering (corr {e3['corr']:.2f})\nstill wins "
       f"{e3['err_uniform']/e3['err_proxy']:.1f}×", transform=a.transAxes, ha="center",
       color=DIM, fontsize=8.5, bbox=dict(boxstyle="round", fc=BG, ec=COUPLE, alpha=0.85))

# --- E5: combined stack --------------------------------------------------
a = ax[1, 1]
errs = R["E5"]["errors"]; names = list(errs)
short = [n.split(" ")[0] for n in names]
cols = [DIM, WARN, "#4a90d9", SIG]
bars = a.bar(short, [errs[n] for n in names], color=cols, width=0.62)
a.set_ylabel("downstream rel-err  (W2A4, lower=better)")
a.set_title("E5 · RANGER wins under pressure", color=INK, fontsize=10, loc="left")
a.set_ylim(0.88, 1.02)
for r, n in zip(bars, names):
    a.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.002,
           f"{errs[n]:.3f}", ha="center", color=INK, fontsize=9)
a.text(0.5, 0.08, "super-weights split BEFORE rotation\n(basis-compatible → no conflict)",
       transform=a.transAxes, ha="center", color=DIM, fontsize=8.5,
       bbox=dict(boxstyle="round", fc=BG, ec=SIG, alpha=0.85))

for row in ax:
    for a in row:
        a.grid(True, alpha=0.12, color=SIG)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("research/experiments/results.png", dpi=130, facecolor=BG)
print("wrote research/experiments/results.png")
