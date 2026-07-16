#!/usr/bin/env python3
"""Render E6 (width x bits compound test) from sweep_results.json."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("research/experiments/sweep_results.json"))
BG, PANEL, INK, DIM = "#0b1116", "#111a22", "#e7eef3", "#9fb2bf"
SIG, WARN, CRIT, COUPLE, BLUE = "#3bc9b0", "#e0a13a", "#e5544e", "#a888f0", "#4a90d9"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor": "#1e2c37", "text.color": INK, "axes.labelcolor": DIM,
    "xtick.color": DIM, "ytick.color": DIM, "font.family": "monospace", "font.size": 9})
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle("E6 · Do WIDTH (P4) and ROTATION (P1) compound against the 4-bit floor?",
             color=INK, fontsize=13, fontweight="bold")

# --- kurtosis overlap ---
a = ax[0, 0]; k = R["kurtosis"]
names = ["raw S", "rotation", "width\nK=8"]
vals = [k["raw"], k["rot"], k["width"]["8"]]
a.bar(names, vals, color=[CRIT, SIG, COUPLE], width=0.6)
a.set_ylabel("excess kurtosis (outlier-ness)")
a.set_title("both mechanisms crush outliers → OVERLAP", color=INK, fontsize=10, loc="left")
for i, v in enumerate(vals): a.text(i, v + 1, f"{v:.0f}", ha="center", color=INK)

# --- dB decomposition ---
a = ax[0, 1]; d = R["db_decomp"]
labs = ["rotation\nonly", "width\nonly", "BOTH"]
gs = [d["g_rot"], d["g_wid"], d["g_both"]]
b = a.bar(labs, gs, color=[SIG, COUPLE, WARN], width=0.6)
a.set_ylabel("error reduction vs raw (dB)")
a.set_title(f"rotation adds only {d['rot_adds_db']:+.1f} dB on top of width",
            color=INK, fontsize=10, loc="left")
for r, v in zip(b, gs): a.text(r.get_x()+r.get_width()/2, v+0.15, f"{v:+.1f}", ha="center", color=INK)
a.annotate("", xy=(2, d["g_both"]), xytext=(1, d["g_wid"]),
           arrowprops=dict(arrowstyle="->", color=CRIT, lw=1.5))
a.text(1.5, d["g_wid"]+0.6, "≈0", color=CRIT, ha="center", fontsize=11, fontweight="bold")

# --- averaging axis (unique to width) ---
a = ax[1, 0]; av = R["averaging"]
Ks = sorted(int(x) for x in av["rows"]); e1 = av["rows"][str(Ks[0])]
ydb = [10*np.log10(e1**2/av["rows"][str(K)]**2) for K in Ks]
a.plot(Ks, ydb, "o-", color=COUPLE, lw=2, ms=7)
a.set_xscale("log", base=2); a.set_xlabel("width factor K")
a.set_ylabel("error reduction, CLEAN signal (dB)")
a.set_title(f"width's UNIQUE axis: averaging {av['slope_db_per_2x']:+.1f} dB/2×",
            color=INK, fontsize=10, loc="left")
a.text(0.05, 0.9, "rotation gives 0 dB here", transform=a.transAxes, color=DIM, fontsize=9,
       bbox=dict(boxstyle="round", fc=BG, ec=SIG, alpha=0.8))

# --- grid heatmap ---
a = ax[1, 1]; g = R["grid"]; Ks2 = g["Ks"]; bits = g["bits"]
M = np.array([g["vals"][str(K)] for K in Ks2])
im = a.imshow(np.log10(M), cmap="viridis_r", aspect="auto")
a.set_xticks(range(len(bits))); a.set_xticklabels([f"{b}b" for b in bits])
a.set_yticks(range(len(Ks2))); a.set_yticklabels([f"K={K}" for K in Ks2])
a.set_title("error(K, b) — outlier signal (log₁₀)", color=INK, fontsize=10, loc="left")
for i in range(len(Ks2)):
    for j in range(len(bits)):
        a.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
               color="white" if M[i,j] > 0.1 else "black", fontsize=8)

for row in ax:
    for a in row: a.grid(True, alpha=0.12, color=SIG)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("research/experiments/sweep.png", dpi=130, facecolor=BG)
print("wrote research/experiments/sweep.png")
