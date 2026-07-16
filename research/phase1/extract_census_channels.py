#!/usr/bin/env python3
"""Extract the top-10 recurring residual-stream channels from an
analyze_real_weights.py census JSON (the weight-only census set that E8's
overlap analysis and E11's exemption arms consume).

Read side: COLUMN indices of super-weights in tensors whose input is the
residual stream (q/k/v/gate/up_proj — shape [out, hidden]). Write side:
ROW indices where the output is the residual stream (o_proj, down_proj —
shape [hidden, in]).

Usage: python extract_census_channels.py <census.json> <hidden_size>
Prints a comma list (for --census-channels) and a per-channel table.
"""
import json
import sys
from collections import Counter

READ_ROLES = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
WRITE_ROLES = ("o_proj", "down_proj")


def main():
    path, hidden = sys.argv[1], int(sys.argv[2])
    d = json.load(open(path))
    hits = Counter()
    read_hits, write_hits = Counter(), Counter()
    for t in d["tensors"]:
        role = t["name"].rsplit(".", 1)[-2].rsplit(".", 1)[-1]
        for sw in t.get("super_weights", []):
            r, c = sw["index"]
            if role in READ_ROLES and t["shape"][1] == hidden:
                hits[c] += 1
                read_hits[c] += 1
            elif role in WRITE_ROLES and t["shape"][0] == hidden:
                hits[r] += 1
                write_hits[r] += 1
    top = hits.most_common(10)
    for ch, n in top:
        print(f"  ch {ch:5d}: {n:3d} hits ({read_hits[ch]} read / "
              f"{write_hits[ch]} write)", file=sys.stderr)
    print(",".join(str(ch) for ch, _ in top))


if __name__ == "__main__":
    main()
