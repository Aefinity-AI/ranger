# Experiment C — quantized-drafter speculative decoding

**The claim under test** (report §2.9): with greedy speculative decoding, a drafted token
is accepted iff it equals the target model's argmax — rejected drafts are simply replaced.
A bad drafter can only cost **speed**, never **quality**. That makes the drafter the one
place in the system where extreme quantization (down to ternary) is **provably risk-free**.

## Falsifiable predictions

- **P-C1** — acceptance degrades slowly down to ~3-bit, then cliffs at 2-bit/ternary
  (PTQ ternary *should* be terrible — ParetoQ says sub-3-bit needs QAT; the curve should
  show exactly that cliff). If acceptance collapses already at 4-bit, the story is wounded.
- **P-C2** — even a heavily quantized drafter nets positive end-to-end speedup vs no
  drafting, until acceptance collapses.
- **P-C3** — outputs are **bit-identical** to plain greedy decoding at every drafter
  precision. This is the exactness guarantee, demonstrated live rather than assumed.

## Run it

```bash
cd research/drafter
source ../phase1/.venv/bin/activate

python drafter_bench.py                                   # 135M drafts for 1.7B (GPU best)
python drafter_bench.py --target HuggingFaceTB/SmolLM2-360M   # CPU-friendly quick run
python drafter_bench.py --selftest                        # network-free pipeline check
```

Per drafter config `{fp, w8, w4, w3, w2, ternary}` it measures:

| Metric | How |
|---|---|
| greedy acceptance | teacher-forced **argmax agreement** on WikiText — mathematically the same thing, measured exactly |
| expected accepted drafts/round | iid estimate `Σ pⁱ` for draft length k=5, plus empirical run lengths |
| end-to-end tok/s | timed `generate(assistant_model=...)` vs the no-drafter baseline |
| exactness | asserts assisted output == plain greedy output, token for token |

Output: console table + `drafter_results.json`.

## Reading the result

- The `exact_output` column should be `YES` in every row — if it ever isn't, P-C3 is
  violated and that's a bug report for transformers, not a quantization finding.
- The interesting curve is `agreement` and `speedup_x` vs bits. RANGER's prediction is a
  knee at ~3 bits. Wherever speedup crosses 1.0× is the honest deployment boundary for a
  PTQ-quantized drafter; pushing the boundary lower (ternary) is what Phase-2 QAT on the
  *drafter* would buy — that's the follow-up if the cliff shows where predicted.

## Notes

- Target and drafter must share a tokenizer (asserted at startup). SmolLM2 sizes do.
- Assisted generation uses batch size 1 (a transformers constraint) — appropriate anyway,
  since edge decode is the memory-bound, batch-1 regime this claim is about.
- transformers' draft-length heuristic adapts k dynamically; we record defaults rather
  than pinning it, since that's the deployment-realistic setting.
- CPU works (use the 360M target); GPU recommended for the 1.7B headline numbers.
