#!/usr/bin/env python3
"""Render the census: kurtosis-vs-damage scatter + per-layer kurtosis profiles."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "census.json"
C = {k: v for k, v in json.load(open(path)).items()
     if not k.startswith("_") and "error" not in v}
if len(C) < 2:
    sys.exit("need >=2 successful models in census.json")

BG, PANEL, INK, DIM = "#0b1116", "#111a22", "#e7eef3", "#9fb2bf"
SIG, CRIT, COUPLE = "#3bc9b0", "#e5544e", "#a888f0"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor": "#1e2c37", "text.color": INK, "axes.labelcolor": DIM,
    "xtick.color": DIM, "ytick.color": DIM, "font.family": "monospace", "font.size": 9})

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Outlier census — does activation kurtosis predict PTQ damage?",
             color=INK, fontsize=12, fontweight="bold")

# scatter: peak kurtosis vs W3 log-damage, marker = QK-Norm
a = ax[0]
for n, r in C.items():
    qkn = r["arch"]["qk_norm"]
    a.scatter(r["act"]["peak_kurtosis"], r["damage"]["w3"],
              s=90, c=SIG if qkn else CRIT, marker="o" if qkn else "s", zorder=3)
    a.annotate(n.split("/")[-1], (r["act"]["peak_kurtosis"], r["damage"]["w3"]),
               textcoords="offset points", xytext=(7, 4), fontsize=8, color=DIM)
a.set_xscale("log")
a.set_xlabel("peak residual-stream excess kurtosis (log)")
a.set_ylabel("W3 log-damage  log(PPL_w3 / PPL_fp)")
a.set_title("each point = one model family", color=INK, fontsize=10, loc="left")
a.scatter([], [], c=SIG, marker="o", label="QK-Norm")
a.scatter([], [], c=CRIT, marker="s", label="no QK-Norm")
a.legend(facecolor=PANEL, edgecolor="#1e2c37", labelcolor=INK)

# per-layer kurtosis profiles (the emergence-layer shape)
a = ax[1]
cmap = plt.cm.viridis(np.linspace(0.25, 0.95, len(C)))
for (n, r), c in zip(C.items(), cmap):
    ks = r["act"]["per_layer_kurtosis"]
    a.plot(np.linspace(0, 1, len(ks)), ks, "-o", ms=3, lw=1.4, color=c,
           label=n.split("/")[-1])
a.set_yscale("symlog")
a.set_xlabel("relative depth")
a.set_ylabel("excess kurtosis (symlog)")
a.set_title("per-layer profiles — the emergence-layer spike", color=INK,
            fontsize=10, loc="left")
a.legend(facecolor=PANEL, edgecolor="#1e2c37", labelcolor=INK, fontsize=7.5)

for x in ax:
    x.grid(True, alpha=0.12, color=SIG)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("census.png", dpi=130, facecolor=BG)
print("wrote census.png")
